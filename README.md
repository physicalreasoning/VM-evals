# VM-evals

Evaluating the internal representations of modern video models against primate neural recordings during an intuitive physics task.

## Overview

We compare hidden-state representations from open-weight video models to neural population recordings from macaque dorsomedial frontal cortex (DMFC) during a Pong ball interception task. The goal is to determine which model architectures and training objectives produce representations that are most structurally similar to biological neural population dynamics during physical prediction.

### Task

The MentalPong dataset consists of 79 Pong ball trajectory conditions. A ball travels rightward across a board, optionally bouncing off a horizontal wall, and is occluded partway through. The monkey must intercept the ball with a paddle at the right edge. Neural recordings (1,889 reliable DMFC neurons, 50ms bins) capture population activity during the occluded epoch, when the monkey must mentally simulate the ball's trajectory.

### Models

- **DINOv2 ViT-B/14** --- Self-supervised image model (per-frame, no temporal context). 768-dimensional features at 4 layers across depth (blocks 2, 5, 8, 11).
- **VideoMAE V2 ViT-Base** --- Masked autoencoder video model. 16-frame input clips with tubelet size 2x16x16, producing 8 temporal tokens per clip. Non-overlapping windows tiled across each trial yield ~88 time bins. 768-dimensional features at encoder layers 3, 6, 9, 11.
- **V-JEPA 2 ViT-L** --- Self-supervised video joint-embedding predictive architecture. 64-frame input, tubelet stride 2, producing 32 temporal tokens with 256 spatial tokens each. 1024-dimensional features at layers 5, 11, 17, 23. Spatial tokens averaged per temporal position.

All models are evaluated using the same battery of analyses applied to the neural data, enabling direct comparison of representational properties.

## Eval Battery

Six representational analyses, each applied identically to neural and model activations:

1. **Eigenspectrum** --- PCA variance explained by top components. Measures how concentrated the representation is along a few axes.
2. **Participation ratio** --- Effective dimensionality of the population state at each time bin. Higher values indicate more distributed representations.
3. **Geometry (PCA/LDA orthogonality)** --- Conditions are classified into 4 groups by bounce count (0 or 1) and initial vertical velocity (up or down). PCA is fit on early time bins, then LDA is applied to the two classification axes independently. The angle between the LDA axes measures whether the two task variables are factorized on orthogonal subspaces. 90 degrees indicates full orthogonality.
4. **Endpoint decoding** --- Cross-validated ridge regression decoding the ball's final y-position from the population state at each time bin. Nested 5-fold outer / 4-fold inner CV with alpha selection.
5. **Full trajectory decoding** --- From a single early population state (250ms), decode the ball's y-position at 29 future time points aligned to interception.
6. **Bounce classification** --- Linear SVM classifying bounce vs. no-bounce from pre-collision population states (250-500ms). Tests whether the XOR-like bounce structure is linearly separable.

Additionally:

7. **RSA (Representational Similarity Analysis)** --- Spearman correlation between Euclidean-distance RDMs of model and neural representations, computed over shared time bins.
8. **Behavioral comparison** --- Cross-validated endpoint predictions from each model are compared to the monkey's actual paddle placement. The correlation between model and monkey signed error patterns measures whether models fail on the same conditions.

## Results

### Main results table

All values report the best layer per model unless otherwise noted.

| Metric | DMFC (1889 neurons) | DMFC FA50 | DINOv2 ViT-B/14 | VideoMAE V2 | V-JEPA 2 ViT-L |
|---|---|---|---|---|---|
| Endpoint decoding (peak r) | 0.933 | 0.924 | 0.951 (blk 11) | 0.970 (L6) | 0.948 (L17) |
| Trajectory decoding (peak r) | 0.391 | 0.415 | 0.260 (blk 2) | 0.609 (L6) | 0.939 (L23) |
| RSA vs DMFC (rho) | --- | --- | 0.188 (blk 11) | 0.016 (L11) | 0.279 (L11) |
| Geometry angle | 56.5 | 82.5 | 90.3 (blk 8) | 78.4 (L11) | 95.3 (L17) |
| Geometry accuracy | 45.6% | 59.5% | 49.4% (blk 8) | 62.0% (L6) | 58.2% (L17) |
| Participation ratio | 9.2 | 32.1 | 10.4 (blk 8) | 19.8 (L9) | 25.2 (L11) |
| Eigenspectrum top 3 | 58.2% | 6.4% | 44.0% (blk 8) | 75.2% (L9) | 79.3% (L5) |
| Bounce classification | 0.480 | 0.573 | 0.557 (blk 8) | 0.533 (L6) | 0.552 (L23) |
| Behavioral error corr | 0.780 | 0.772 | 0.771 (blk 11) | 0.775 (L6) | 0.781 (L17) |

### Per-layer results

| System | Eigen top 3 | PR | Angle | Geo acc | Endpoint r | Traj r | Bounce acc | RSA rho | Error corr |
|---|---|---|---|---|---|---|---|---|---|
| DMFC raw | 58.2% | 9.2 | 56.5 | 45.6% | 0.933 | 0.391 | 0.480 | --- | 0.780 |
| DMFC FA50 | 6.4% | 32.1 | 82.5 | 59.5% | 0.924 | 0.415 | 0.573 | --- | 0.772 |
| DINOv2 blk 2 | 55.4% | 4.9 | 22.7 | 43.0% | 0.605 | 0.260 | 0.482 | 0.003 | 0.602 |
| DINOv2 blk 5 | 47.7% | 8.3 | 105.9 | 43.0% | 0.924 | 0.080 | 0.565 | 0.059 | 0.726 |
| DINOv2 blk 8 | 44.0% | 10.4 | 90.3 | 49.4% | 0.942 | 0.239 | 0.557 | 0.077 | 0.723 |
| DINOv2 blk 11 | 60.2% | 6.9 | 37.0 | 45.6% | 0.951 | 0.108 | 0.467 | 0.188 | 0.771 |
| VideoMAE2 L3 | 63.0% | 15.9 | 35.6 | 57.0% | 0.958 | 0.494 | 0.379 | -0.008 | 0.761 |
| VideoMAE2 L6 | 65.7% | 13.7 | 71.8 | 62.0% | 0.970 | 0.609 | 0.533 | -0.006 | 0.775 |
| VideoMAE2 L9 | 75.2% | 19.8 | 52.0 | 62.0% | 0.962 | 0.576 | 0.522 | 0.008 | 0.759 |
| VideoMAE2 L11 | 62.1% | 17.8 | 78.4 | 60.8% | 0.962 | 0.598 | 0.472 | 0.016 | 0.719 |
| V-JEPA 2 L5 | 79.3% | 7.0 | 99.9 | 45.6% | 0.572 | 0.486 | 0.423 | 0.070 | 0.535 |
| V-JEPA 2 L11 | 76.7% | 25.2 | 37.2 | 49.4% | 0.919 | 0.756 | 0.445 | 0.279 | 0.701 |
| V-JEPA 2 L17 | 71.6% | 14.9 | 95.3 | 58.2% | 0.948 | 0.908 | 0.508 | 0.194 | 0.781 |
| V-JEPA 2 L23 | 60.2% | 14.1 | 46.9 | 57.0% | 0.944 | 0.939 | 0.552 | 0.150 | 0.738 |

### Summary of findings

**Trajectory decoding separates temporal from non-temporal models.** V-JEPA 2 layer 23 achieves r = 0.939 on trajectory decoding, substantially exceeding both VideoMAE V2 (0.609) and DINOv2 (0.260). DINOv2 processes frames independently and cannot represent trajectory dynamics; VideoMAE V2 operates over short 16-frame clips; V-JEPA 2 integrates 64 frames. DMFC itself achieves 0.391, suggesting that the neural code encodes trajectories in a form that is not easily decoded by linear methods from single time-bin snapshots.

**V-JEPA 2 produces the most brain-like representations.** V-JEPA 2 layer 11 achieves the highest RSA correlation with DMFC (rho = 0.279). VideoMAE V2 RSA is near zero across all layers (rho ~ 0.01), indicating that despite strong decoding performance, its internal geometry bears no resemblance to DMFC. DINOv2 block 11 falls between the two (rho = 0.188). This dissociation --- high task performance without representational similarity --- suggests that the training objective, not merely the capacity to represent physics, determines alignment with biological representations.

**Orthogonal factorization of task variables.** The geometry analysis reveals near-orthogonal separation of bounce/no-bounce and launch direction axes in V-JEPA 2 layer 17 (95.3 degrees), DINOv2 block 8 (90.3 degrees), and DMFC FA50 (82.5 degrees). This factorized geometry emerges in the mid-to-late layers of both model families and in the low-dimensional neural manifold, consistent with an abstract coding scheme for independent task variables.

**Endpoint decoding saturates across all systems.** All deep model layers and DMFC achieve r > 0.92 for endpoint decoding. This is expected given that the ball's endpoint is a low-dimensional variable that can be read out from any representation containing position information. VideoMAE V2 layer 6 achieves the highest value (0.970).

**Bounce classification is near chance everywhere.** No system --- neural or model --- achieves above-chance linear classification of bounce vs. no-bounce from pre-collision states (all accuracies 0.38--0.57, all p > 0.1). This is consistent with the XOR-like structure of the bounce variable: it depends on the interaction of vertical velocity and wall proximity, which is not linearly separable from early states alone.

**All models produce monkey-like error patterns.** The behavioral comparison reveals that all systems, when decoding endpoints, make errors correlated with the monkey's interception errors (r = 0.53--0.78). V-JEPA 2 layer 17 achieves the highest correlation (0.781), slightly exceeding DMFC raw (0.780). The monkey shows a systematic positive bias of 0.36 units on bounce conditions; V-JEPA 2 layer 17 most closely matches this bias (0.33).

**Dimensionality tracks with temporal context.** V-JEPA 2 layer 11 has a participation ratio of 25.2, approaching DMFC FA50 (32.1). VideoMAE V2 layers occupy an intermediate range (14--20). DINOv2 is lowest (5--10). Models with broader temporal integration tend to produce higher-dimensional representations, consistent with the idea that trajectory prediction requires distributing information across more dimensions.

## Data

The neural dataset is from the MentalPong study, available at [Zenodo](https://zenodo.org/records/13952210). It contains:

- Trial-averaged DMFC neural responses (1,889 reliable neurons, 79 conditions, 50ms bins)
- Factor analysis latent representations (FA10, FA20, FA50)
- Behavioral responses including ball trajectories, paddle positions, and eye movements
- Condition metadata (initial positions, velocities, bounce counts)

Rendered Pong stimuli (224x224 RGB frames, 20 fps) are generated from the TRUE ball positions in the dataset using `src/utils/pong_renderer.py`.

## Usage

All GPU computation runs on [Modal](https://modal.com). No local GPU is required.

```bash
# 1. Upload rendered frames + neural data to Modal volume (one-time)
python -m src.models.modal_extract upload

# 2. Run feature extraction + eval battery on Modal GPU
modal run src/models/modal_extract.py

# 3. Download results
python -m src.models.modal_extract download
```

### Rendering stimuli

```bash
python -m src.utils.pong_renderer
```

### Project structure

```
src/
  evals/
    dimensionality.py   # Eigenspectrum + participation ratio
    geometry.py          # PCA/LDA orthogonality analysis
    decoding.py          # Endpoint + trajectory decoding (nested CV ridge)
    bounce_classification.py  # Linear SVM XOR test
    rsa.py               # RDM computation + Spearman correlation
    behavioral.py        # Model vs monkey error comparison
  models/
    modal_extract.py     # Modal pipeline: extraction + eval battery
    feature_extractor.py # Local feature extraction utilities
  utils/
    data_loader.py       # Dataset loading + preprocessing
    pong_renderer.py     # Pong stimulus renderer
data/
  zenodo/               # Neural dataset (not tracked)
  rendered/             # Rendered frames (not tracked)
results/
  eval_results.json     # Full results
```
