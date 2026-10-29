"""
Main pipeline: run the full eval battery on video model representations.

Usage:
    python -m src.run_eval_battery --model dinov2_video --data-dir data/zenodo

Steps:
    1. Load DMFC neural data + 79 condition metadata from Zenodo dataset
    2. Load or extract video model features for the 79 conditions
    3. Run all 5 evals: dimensionality, geometry, endpoint decoding,
       full trajectory decoding, bounce classification, RSA
    4. Save results as JSON + generate comparison plots
"""

import argparse
import json
import numpy as np
from pathlib import Path

from src.utils.data_loader import (
    load_neural_dataset,
    extract_initial_conditions,
    extract_neural_data,
    get_condition_metadata,
)
from src.evals.dimensionality import eigenspectrum, participation_ratio
from src.evals.geometry import geometry_analysis, classify_conditions
from src.evals.decoding import endpoint_decoding, full_trajectory_decoding
from src.evals.bounce_classification import bounce_classification
from src.evals.rsa import compute_rdm, rsa_correlation


def run_battery(states: np.ndarray, system_name: str,
                endpoint_y: np.ndarray, ball_positions_y: np.ndarray,
                trial_durations: np.ndarray,
                n_bounces: np.ndarray, vel_y0: np.ndarray) -> dict:
    """
    Run the full eval battery on a single system's states.

    Args:
        states: (N_units, 79, T) population states
        system_name: label for this system
        endpoint_y: (79,) ball endpoint y-positions
        ball_positions_y: (79, T) ball y-positions over time
        trial_durations: (79,) number of valid time bins per condition
        n_bounces: (79,) bounce counts
        vel_y0: (79,) initial y-velocities

    Returns:
        dict of results keyed by eval name
    """
    print(f"\n{'='*60}")
    print(f"Running eval battery on: {system_name}")
    print(f"States shape: {states.shape}")
    print(f"{'='*60}")

    results = {"system": system_name, "shape": list(states.shape)}

    # 1. Eigenspectrum
    print("  [1/5] Eigenspectrum...")
    eigen = eigenspectrum(states)
    results["eigenspectrum"] = {
        "variance_explained": eigen["variance_explained"].tolist(),
        "shuffle_mean": eigen["shuffle_mean"].tolist(),
    }
    print(f"    Top 3 PCs explain: {eigen['variance_explained'][:3].sum():.1f}%")

    # 2. Participation ratio
    print("  [2/5] Participation ratio...")
    pr = participation_ratio(states)
    results["participation_ratio"] = {
        "pr": pr["pr"].tolist(),
        "time_ms": pr["time_ms"].tolist(),
    }
    stable_pr = pr["pr"][~np.isnan(pr["pr"])]
    if len(stable_pr) > 0:
        print(f"    Mean PR: {stable_pr.mean():.1f}")

    # 3. Geometry
    print("  [3/5] Geometry analysis...")
    labels = classify_conditions(n_bounces, vel_y0)
    geo = geometry_analysis(states, labels)
    results["geometry"] = {
        "best_angle": float(geo["best_angle"]),
        "best_accuracy": float(geo["best_accuracy"]),
        "best_time_bin": int(geo["best_time_bin"]),
    }
    print(f"    Angle: {geo['best_angle']:.0f}° (90°=orthogonal)")
    print(f"    4-way accuracy: {geo['best_accuracy']:.0%}")

    # 4. Endpoint decoding
    print("  [4/6] Endpoint decoding...")
    dec = endpoint_decoding(states, endpoint_y)
    results["endpoint_decoding"] = {
        "correlation": dec["correlation"].tolist(),
    }
    peak_r = np.nanmax(dec["correlation"])
    print(f"    Peak correlation: r={peak_r:.3f}")

    # 5. Full trajectory decoding
    print("  [5/6] Full trajectory decoding from 250ms...")
    traj = full_trajectory_decoding(states, ball_positions_y,
                                    trial_durations, decode_from_time=5)
    results["trajectory_decoding"] = {
        "correlation_vs_time": traj["correlation_vs_time"].tolist(),
        "time_relative_ms": traj["time_relative_ms"].tolist(),
    }
    peak_traj_r = np.nanmax(traj["correlation_vs_time"])
    print(f"    Peak trajectory correlation: r={peak_traj_r:.3f}")

    # 6. Bounce classification
    print("  [6/6] Bounce classification...")
    bc = bounce_classification(states, n_bounces)
    results["bounce_classification"] = {
        "balanced_accuracy": float(bc["balanced_accuracy"]),
        "p_value": float(bc["p_value"]),
    }
    print(f"    Balanced accuracy: {bc['balanced_accuracy']:.3f}")
    print(f"    p-value: {bc['p_value']:.4f}")

    return results


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load neural data
    print("Loading DMFC neural data...")
    dat = load_neural_dataset(data_dir=data_dir)

    ic = extract_initial_conditions(dat)
    neural = extract_neural_data(dat, epoch="occ")
    meta = get_condition_metadata(dat)

    neural_states = neural["firing_rates"]  # (N_neurons, 79, T) trimmed
    trimmed_bins = neural.get("trimmed_bins", 0)
    endpoint_y = meta["ball_endpoint_y"]
    n_bounces = meta["n_bounces"]
    trial_durations = meta["trial_duration"]
    vel_y0 = ic["vel_y0"]

    # Ball positions for trajectory decoding (visible + occluded, TRUE positions)
    # Trim to match neural data alignment
    ball_pos_y = np.concatenate([
        dat["behavioral_responses"]["vis"]["ball_pos_y_TRUE"],
        dat["behavioral_responses"]["occ"]["ball_pos_y_TRUE"]
    ], axis=1)
    if trimmed_bins > 0:
        ball_pos_y = ball_pos_y[:, trimmed_bins:]
        trial_durations = np.clip(trial_durations - trimmed_bins, 1,
                                  ball_pos_y.shape[1])

    print(f"Neural data: {neural_states.shape}")
    print(f"Conditions: {len(endpoint_y)}")
    print(f"Bounce conditions: {(n_bounces > 0).sum()}")
    print(f"No-bounce conditions: {(n_bounces == 0).sum()}")

    all_results = {}

    # Run on DMFC neural data (raw 1889 neurons)
    neural_results = run_battery(
        neural_states, "DMFC_raw",
        endpoint_y, ball_pos_y, trial_durations, n_bounces, vel_y0
    )
    all_results["DMFC_raw"] = neural_results

    # Run on DMFC FA50 representation (50 latent factors)
    fa_key = "neural_responses_reliable_FactorAnalysis_50"
    if fa_key in dat:
        fa50_states = dat[fa_key]["occ"]
        # Trim leading NaN bins from FA50 data
        fa50_first_valid = 0
        for t in range(fa50_states.shape[2]):
            if not np.all(np.isnan(fa50_states[:, :, t])):
                fa50_first_valid = t
                break
        if fa50_first_valid > 0:
            fa50_states = fa50_states[:, :, fa50_first_valid:]
        fa50_results = run_battery(
            fa50_states, "DMFC_FA50",
            endpoint_y, ball_pos_y, trial_durations, n_bounces, vel_y0
        )
        all_results["DMFC_FA50"] = fa50_results

    # Run on video model
    if args.model:
        print(f"\nLoading model features: {args.model}")
        model_features_path = (
            output_dir / f"{args.model}_features.npz"
        )

        if model_features_path.exists():
            print(f"Loading cached features from {model_features_path}")
            model_data = np.load(model_features_path, allow_pickle=True)
            model_states_dict = model_data["features"][()]
        else:
            print("Features not cached. Run feature extraction first.")
            print("See src/models/feature_extractor.py")
            return

        for layer_name, model_states in model_states_dict.items():
            name = f"{args.model}/{layer_name}"
            model_results = run_battery(
                model_states, name,
                endpoint_y, ball_pos_y, trial_durations, n_bounces, vel_y0
            )
            all_results[name] = model_results

        # RSA comparison
        print("\nRunning RSA comparison...")
        rdm_neural = compute_rdm(neural_states)
        for layer_name, model_states in model_states_dict.items():
            rdm_model = compute_rdm(model_states)
            rsa_result = rsa_correlation(rdm_neural, rdm_model)
            name = f"{args.model}/{layer_name}"
            all_results[name]["rsa_vs_dmfc"] = {
                "rho": float(rsa_result["rho"]),
                "p_value": float(rsa_result["p_value"]),
            }
            print(f"  {name}: rho={rsa_result['rho']:.3f}")

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run eval battery on video model representations"
    )
    parser.add_argument("--data-dir", type=str, default="data/zenodo",
                        help="Path to Zenodo dataset directory")
    parser.add_argument("--model", type=str, default=None,
                        help="Video model name (e.g., dinov2_video)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()
    main(args)
