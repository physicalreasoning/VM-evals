"""
Modal-based GPU feature extraction + eval battery.

Usage:
    # Upload data (one-time):
    python -m src.models.modal_extract upload

    # Run extraction + evals on Modal GPU:
    modal run src/models/modal_extract.py

    # Download results:
    python -m src.models.modal_extract download
"""

import modal
import sys

app = modal.App("pong-eval-pipeline")

vol = modal.Volume.from_name("pong-data", create_if_missing=True)
VOL_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "torchvision>=0.16",
        "numpy<2",
        "scikit-learn",
        "scipy",
        "pandas<2",
        "transformers>=4.52",
        "decord",
        "av",
    )
    .add_local_dir("src", remote_path="/root/src")
)


@app.function(
    image=image,
    gpu="A100",
    volumes={VOL_PATH: vol},
    timeout=3600,
)
def run_pipeline():
    """Extract DINOv2 features + run full eval battery on Modal GPU."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import json
    import os
    import sys

    sys.path.insert(0, "/root")

    from src.utils.data_loader import (
        load_neural_dataset, extract_initial_conditions,
        extract_neural_data, get_condition_metadata,
    )
    from src.evals.dimensionality import eigenspectrum, participation_ratio
    from src.evals.geometry import geometry_analysis, classify_conditions
    from src.evals.decoding import endpoint_decoding, full_trajectory_decoding
    from src.evals.bounce_classification import bounce_classification
    from src.evals.rsa import compute_rdm, rsa_correlation
    from src.evals.behavioral import behavioral_comparison

    np.random.seed(42)

    # ── 1. Load neural data ──────────────────────────────────────
    print("Loading DMFC neural data...")
    dat = load_neural_dataset(data_dir=f"{VOL_PATH}/neural")
    ic = extract_initial_conditions(dat)
    neural = extract_neural_data(dat, epoch="occ")
    meta = get_condition_metadata(dat)

    neural_states = neural["firing_rates"]
    trimmed = neural.get("trimmed_bins", 0)
    endpoint_y = meta["ball_endpoint_y"]
    n_bounces = meta["n_bounces"]
    vel_y0 = ic["vel_y0"]
    labels = classify_conditions(n_bounces, vel_y0)
    monkey_paddle_y = dat["meta"]["paddle_y"].values

    # Ball positions for trajectory decoding
    ball_pos_y = np.concatenate([
        dat["behavioral_responses"]["vis"]["ball_pos_y_TRUE"],
        dat["behavioral_responses"]["occ"]["ball_pos_y_TRUE"]
    ], axis=1)
    if trimmed > 0:
        ball_pos_y = ball_pos_y[:, trimmed:]

    n_conds = ball_pos_y.shape[0]
    trial_dur = np.zeros(n_conds, dtype=int)
    for i in range(n_conds):
        valid = np.isfinite(ball_pos_y[i])
        if valid.any():
            trial_dur[i] = np.max(np.where(valid)[0]) + 1

    # FA50 representation
    fa50_states = dat["neural_responses_reliable_FactorAnalysis_50"]["occ"]
    fa50_first_valid = 0
    for t in range(fa50_states.shape[2]):
        if not np.all(np.isnan(fa50_states[:, :, t])):
            fa50_first_valid = t
            break
    if fa50_first_valid > 0:
        fa50_states = fa50_states[:, :, fa50_first_valid:]

    print(f"Neural: {neural_states.shape}, FA50: {fa50_states.shape}")
    print(f"Bounce: {(n_bounces > 0).sum()}, No-bounce: {(n_bounces == 0).sum()}")

    # ── 2. Extract DINOv2 features ───────────────────────────────
    print("\nLoading rendered frames...")
    n_frames_arr = np.load(f"{VOL_PATH}/rendered/n_frames.npy")
    frames_list = []
    for i in range(79):
        frames = np.load(f"{VOL_PATH}/rendered/cond_{i:02d}_full.npy")
        frames_list.append(frames)
    print(f"Loaded {len(frames_list)} conditions, total frames: {sum(len(f) for f in frames_list)}")

    # Check if features already extracted
    feat_path = f"{VOL_PATH}/features/dinov2_features.npz"
    if os.path.exists(feat_path):
        print(f"Loading cached DINOv2 features from {feat_path}")
        dino_data = np.load(feat_path)
        dino_features = {k: dino_data[k] for k in dino_data.keys()}
    else:
        print("Extracting DINOv2 features...")
        dino_features = _extract_dinov2(frames_list)
        os.makedirs(f"{VOL_PATH}/features", exist_ok=True)
        np.savez_compressed(feat_path, **dino_features)
        vol.commit()

    # ── 2b. Extract VideoMAE V2 features ───────────────────────
    vmae_feat_path = f"{VOL_PATH}/features/videomae2_features.npz"
    if os.path.exists(vmae_feat_path):
        print(f"\nLoading cached VideoMAE V2 features from {vmae_feat_path}")
        vmae_data = np.load(vmae_feat_path)
        vmae_features = {k: vmae_data[k] for k in vmae_data.keys()}
    else:
        print("\nExtracting VideoMAE V2 features...")
        vmae_features = _extract_videomae_v2(frames_list)
        os.makedirs(f"{VOL_PATH}/features", exist_ok=True)
        np.savez_compressed(vmae_feat_path, **vmae_features)
        vol.commit()

    # ── 2c. Extract V-JEPA 2 features ──────────────────────────
    vjepa_feat_path = f"{VOL_PATH}/features/vjepa2_features.npz"
    if os.path.exists(vjepa_feat_path):
        print(f"\nLoading cached V-JEPA 2 features from {vjepa_feat_path}")
        vjepa_data = np.load(vjepa_feat_path)
        vjepa_features = {k: vjepa_data[k] for k in vjepa_data.keys()}
    else:
        print("\nExtracting V-JEPA 2 features...")
        vjepa_features = _extract_vjepa2(frames_list)
        os.makedirs(f"{VOL_PATH}/features", exist_ok=True)
        np.savez_compressed(vjepa_feat_path, **vjepa_features)
        vol.commit()

    # ── 3. Run eval battery on all systems ───────────────────────
    all_results = {}

    systems = {
        "DMFC_raw": neural_states,
        "DMFC_FA50": fa50_states,
    }
    for layer_name, feat in dino_features.items():
        systems[f"DINOv2/{layer_name}"] = feat
    for layer_name, feat in vmae_features.items():
        systems[f"VideoMAE2/{layer_name}"] = feat
    for layer_name, feat in vjepa_features.items():
        systems[f"VJEPA2/{layer_name}"] = feat

    for sys_name, states in systems.items():
        print(f"\n{'='*60}")
        print(f"Eval: {sys_name} — shape={states.shape}")
        print(f"{'='*60}")

        results = {"system": sys_name, "shape": list(states.shape)}

        # Eigenspectrum
        eigen = eigenspectrum(states, n_components=20)
        results["eigenspectrum_top3"] = float(eigen["variance_explained"][:3].sum())
        print(f"  Eigenspectrum top 3: {results['eigenspectrum_top3']:.1f}%")

        # Participation ratio
        pr = participation_ratio(states)
        stable = pr["pr"][~np.isnan(pr["pr"])]
        results["participation_ratio_mean"] = float(stable.mean()) if len(stable) > 0 else float("nan")
        print(f"  PR mean: {results['participation_ratio_mean']:.1f}")

        # Geometry
        geo = geometry_analysis(states, labels, n_early_bins=6, max_time_bin=10)
        results["geometry_angle"] = float(geo["best_angle"])
        results["geometry_accuracy"] = float(geo["best_accuracy"])
        print(f"  Geometry: angle={geo['best_angle']:.1f}°, acc={geo['best_accuracy']:.1%}")

        # Endpoint decoding
        dec = endpoint_decoding(states, endpoint_y)
        results["endpoint_peak_r"] = float(np.nanmax(dec["correlation"]))
        print(f"  Endpoint: peak r={results['endpoint_peak_r']:.3f}")

        # Trajectory decoding
        traj = full_trajectory_decoding(states, ball_pos_y, trial_dur, decode_from_time=10)
        results["trajectory_peak_r"] = float(np.nanmax(traj["correlation_vs_time"]))
        print(f"  Trajectory: peak r={results['trajectory_peak_r']:.3f}")

        # Bounce classification
        bc = bounce_classification(states, n_bounces, time_window=(5, 10),
                                   n_repeats=50, n_permutations=50)
        results["bounce_accuracy"] = float(bc["balanced_accuracy"])
        results["bounce_p_value"] = float(bc["p_value"])
        print(f"  Bounce: acc={bc['balanced_accuracy']:.3f}, p={bc['p_value']:.3f}")

        # Behavioral comparison
        beh = behavioral_comparison(states, endpoint_y, monkey_paddle_y, n_bounces)
        results["monkey_accuracy"] = beh["monkey_accuracy"]
        results["behavioral_error_corr"] = beh["best_error_corr"]
        results["behavioral_error_corr_t"] = beh["best_error_corr_t"]
        results["bounce_bias_monkey"] = beh["bounce_bias_monkey"]
        results["bounce_bias_model_peak"] = float(
            beh["bounce_bias_model"][beh["best_error_corr_t"]]
        ) if np.isfinite(beh["bounce_bias_model"][beh["best_error_corr_t"]]) else float("nan")
        print(f"  Behavioral: error_corr={beh['best_error_corr']:.3f} (t={beh['best_error_corr_t']})")

        all_results[sys_name] = results

    # ── 4. RSA: model layers vs DMFC ────────────────────────────
    print("\n" + "="*60)
    print("RSA: model layers vs DMFC")
    print("="*60)

    T_neural = neural_states.shape[2]
    rdm_neural = compute_rdm(neural_states)

    model_feature_sets = [
        ("DINOv2", dino_features),
        ("VideoMAE2", vmae_features),
        ("VJEPA2", vjepa_features),
    ]
    for model_name, feat_dict in model_feature_sets:
        for layer_name, feat in feat_dict.items():
            # Truncate model features to match neural time bins for RSA
            T_model = feat.shape[2]
            T_rsa = min(T_neural, T_model)
            feat_truncated = feat[:, :, :T_rsa]
            neural_truncated = neural_states[:, :, :T_rsa]
            rdm_model = compute_rdm(feat_truncated)
            rdm_neural_trunc = compute_rdm(neural_truncated)
            rsa = rsa_correlation(rdm_neural_trunc, rdm_model)
            key = f"{model_name}/{layer_name}"
            all_results[key]["rsa_vs_dmfc_rho"] = float(rsa["rho"])
            all_results[key]["rsa_vs_dmfc_p"] = float(rsa["p_value"])
            print(f"  {key}: rho={rsa['rho']:.4f}")

    # ── 5. Save results ──────────────────────────────────────────
    os.makedirs(f"{VOL_PATH}/results", exist_ok=True)
    results_path = f"{VOL_PATH}/results/eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    vol.commit()

    return all_results


def _extract_dinov2(frames_list):
    """Extract DINOv2 features from all conditions."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.cuda().eval()
    n_blocks = len(model.blocks)
    layer_names = [
        f"blocks.{n_blocks // 4 - 1}",
        f"blocks.{n_blocks // 2 - 1}",
        f"blocks.{3 * n_blocks // 4 - 1}",
        f"blocks.{n_blocks - 1}",
    ]
    print(f"  Layers: {layer_names}")

    features_store = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            features_store[name] = output.detach().cpu()
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    max_T = max(len(f) for f in frames_list)
    all_features = {}
    batch_size = 64

    for i in range(len(frames_list)):
        if i % 10 == 0:
            print(f"  Condition {i}/{len(frames_list)}...")

        frames = frames_list[i]
        T_i = len(frames)
        cond_feats = {name: [] for name in layer_names}

        for t_start in range(0, T_i, batch_size):
            t_end = min(t_start + batch_size, T_i)
            batch = frames[t_start:t_end]

            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().cuda() / 255.0
            if tensor.shape[-1] != 224 or tensor.shape[-2] != 224:
                tensor = F.interpolate(tensor, size=(224, 224),
                                       mode="bilinear", align_corners=False)
            tensor = (tensor - mean) / std

            features_store = {}
            with torch.no_grad():
                model(tensor)

            for name in layer_names:
                if name in features_store:
                    feat = features_store[name]
                    if feat.ndim == 3:
                        feat = feat[:, 0, :]  # CLS token
                    cond_feats[name].append(feat.numpy())

        for name in layer_names:
            if cond_feats[name]:
                feat_arr = np.concatenate(cond_feats[name], axis=0)
                D = feat_arr.shape[1]
                if name not in all_features:
                    all_features[name] = np.full((D, len(frames_list), max_T), np.nan)
                all_features[name][:, i, :T_i] = feat_arr.T

    for h in hooks:
        h.remove()

    return all_features


def _extract_videomae_v2(frames_list):
    """Extract VideoMAE V2 features from all conditions.

    VideoMAE V2 takes 16 frames → 8 temporal tokens (tubelet stride=2).
    We slide non-overlapping 16-frame windows across each condition's frames,
    spatially average tokens at each temporal position, and concatenate
    across windows. ~186 frames / 16 = ~11 windows × 8 = ~88 time bins.
    Output shape: (D, 79, max_T_out) per layer.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import VideoMAEModel, VideoMAEImageProcessor

    model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
    print(f"  Loading VideoMAE V2: {model_id}")
    processor = VideoMAEImageProcessor.from_pretrained(model_id)
    model = VideoMAEModel.from_pretrained(model_id).cuda().eval()

    # ViT-Base has 12 encoder layers
    layer_indices = [3, 6, 9, 11]
    layer_names = [f"encoder.layer.{l}" for l in layer_indices]
    print(f"  Layers: {layer_indices}")

    CLIP_LEN = 16
    T_PER_CLIP = 8  # 16 frames / tubelet temporal stride 2
    n_conds = len(frames_list)

    # First pass: compute max output time bins across conditions
    max_T_out = 0
    for frames in frames_list:
        n_clips = max(1, len(frames) // CLIP_LEN)
        max_T_out = max(max_T_out, n_clips * T_PER_CLIP)

    all_features = {}

    for i in range(n_conds):
        if i % 10 == 0:
            print(f"  Condition {i}/{n_conds}...")

        frames = frames_list[i]  # (T_i, H, W, 3) uint8
        T_i = len(frames)
        n_clips = max(1, T_i // CLIP_LEN)

        cond_feats = {name: [] for name in layer_names}

        for c in range(n_clips):
            start = c * CLIP_LEN
            end = start + CLIP_LEN
            if end > T_i:
                # Pad last clip by repeating final frame
                clip = np.concatenate([
                    frames[start:T_i],
                    np.stack([frames[-1]] * (end - T_i))
                ], axis=0)
            else:
                clip = frames[start:end]

            # Processor expects list of (H, W, 3) arrays
            frame_list = [clip[t] for t in range(CLIP_LEN)]
            inputs = processor(frame_list, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # outputs.hidden_states: tuple of 13 tensors (embed + 12 layers)
            # each (1, n_tokens, 768)
            # n_tokens = 8 temporal * 196 spatial = 1568 for 224px
            for li, layer_idx in enumerate(layer_indices):
                # +1 because index 0 is the patch embedding
                feat = outputs.hidden_states[layer_idx + 1][0].cpu()  # (1568, D)
                D = feat.shape[-1]
                n_tokens = feat.shape[0]
                n_spatial = n_tokens // T_PER_CLIP  # 196

                # Reshape to (T_PER_CLIP, n_spatial, D), average spatial
                feat = feat.view(T_PER_CLIP, n_spatial, D).mean(dim=1)  # (8, D)
                cond_feats[layer_names[li]].append(feat.numpy())

        # Concatenate clips → (n_clips * 8, D)
        for name in layer_names:
            if cond_feats[name]:
                feat_arr = np.concatenate(cond_feats[name], axis=0)  # (T_out_i, D)
                D = feat_arr.shape[1]
                T_out_i = feat_arr.shape[0]

                if name not in all_features:
                    all_features[name] = np.full((D, n_conds, max_T_out), np.nan)
                all_features[name][:, i, :T_out_i] = feat_arr.T

    return all_features


def _extract_vjepa2(frames_list):
    """Extract V-JEPA 2 features from all conditions.

    V-JEPA 2 takes 64 frames → 32 temporal tokens (tubelet stride=2).
    We uniformly sample 64 frames from each condition, extract hierarchical
    features at 4 layers, and spatially average to get (32, D) per condition.
    Output shape: (D, 79, 32) per layer.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoVideoProcessor, AutoModel

    model_id = "facebook/vjepa2-vitl-fpc64-256"
    print(f"  Loading V-JEPA 2: {model_id}")
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).cuda().eval()

    # ViT-L has 24 blocks — sample 4 layers across depth
    layer_indices = [5, 11, 17, 23]
    layer_names = [f"layer_{l}" for l in layer_indices]
    print(f"  Layers: {layer_indices}")

    n_conds = len(frames_list)
    T_out = 32  # 64 input frames / tubelet temporal stride 2
    all_features = {}

    for i in range(n_conds):
        if i % 10 == 0:
            print(f"  Condition {i}/{n_conds}...")

        frames = frames_list[i]  # (T_i, H, W, 3) uint8
        T_i = len(frames)

        # Uniformly sample 64 frames
        if T_i >= 64:
            indices = np.linspace(0, T_i - 1, 64, dtype=int)
        else:
            # Repeat last frame to pad to 64
            indices = np.concatenate([
                np.arange(T_i),
                np.full(64 - T_i, T_i - 1, dtype=int)
            ])
        sampled = frames[indices]  # (64, H, W, 3)

        # Processor expects list of frames as list of numpy arrays
        frame_list = [sampled[t] for t in range(64)]
        inputs = processor(frame_list, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of 25 tensors (embed + 24 layers)
        # each (1, 8192, 1024), where 8192 = 32 temporal * 256 spatial
        for li, layer_idx in enumerate(layer_indices):
            feat = outputs.hidden_states[layer_idx][0].cpu()  # (8192, D)
            D = feat.shape[-1]
            n_tokens = feat.shape[0]
            n_spatial = n_tokens // T_out  # 256

            # Reshape to (T_out, n_spatial, D), average over spatial
            feat = feat.view(T_out, n_spatial, D).mean(dim=1)  # (T_out, D)
            feat = feat.numpy()  # (32, D)

            name = layer_names[li]
            if name not in all_features:
                all_features[name] = np.full((D, n_conds, T_out), np.nan)
            all_features[name][:, i, :] = feat.T

    return all_features


@app.local_entrypoint()
def main():
    result = run_pipeline.remote()
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for sys_name, res in result.items():
        print(f"\n{sys_name}:")
        for k, v in res.items():
            if k not in ("system", "shape"):
                print(f"  {k}: {v}")


# --- Local CLI for upload/download ---
def upload_data():
    """Upload rendered frames + neural data to Modal volume."""
    import subprocess
    subprocess.run([
        "modal", "volume", "put", "pong-data",
        "data/rendered", "/rendered",
    ], check=True)
    print("Frames uploaded!")


def download_results():
    """Download results from Modal volume."""
    import subprocess
    import os
    os.makedirs("results", exist_ok=True)
    subprocess.run([
        "modal", "volume", "get", "pong-data",
        "/results", "results",
    ], check=True)
    # Also grab features
    os.makedirs("data/features", exist_ok=True)
    subprocess.run([
        "modal", "volume", "get", "pong-data",
        "/features", "data/features",
    ], check=True)
    print("Download complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "upload":
            upload_data()
        elif cmd == "download":
            download_results()
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("Usage:")
        print("  python -m src.models.modal_extract upload    # Upload data")
        print("  modal run src/models/modal_extract.py        # Run on GPU")
        print("  python -m src.models.modal_extract download  # Download results")
