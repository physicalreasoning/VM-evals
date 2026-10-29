"""
Data loader for the MentalPong Zenodo dataset.

Loads the 79 Pong conditions (initial ball positions and velocities)
and the DMFC neural data (1,889 neurons, 50ms bins).

Dataset: https://zenodo.org/records/13952210
"""

import numpy as np
import pickle as pk
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Compatibility shim: the pickled dataset was created with an older pandas
# that had pandas.core.indexes.numeric (removed in pandas 2.0).
import pandas.core.indexes
if not hasattr(pandas.core.indexes, "numeric"):
    import types
    mod = types.ModuleType("pandas.core.indexes.numeric")
    mod.Int64Index = getattr(pandas, "Index", None)
    mod.Float64Index = getattr(pandas, "Index", None)
    mod.UInt64Index = getattr(pandas, "Index", None)
    sys.modules["pandas.core.indexes.numeric"] = mod
    pandas.core.indexes.numeric = mod

# The 79 condition indices from the MentalPong codebase
PONG_BASIC_META_IDX = [
    29337, 55062, 58244, 59920, 68220, 72780, 77046, 77944, 90751,
    93053, 99902, 105802, 118879, 122957, 126479, 158705, 159613, 163248,
    167848, 179553, 182748, 183171, 184642, 187617, 197632, 199564, 204372,
    204678, 206128, 217817, 218791, 226533, 232533, 241919, 242380, 244437,
    245662, 248856, 251629, 258471, 269179, 270569, 273073, 273515, 287936,
    297652, 301600, 320246, 325813, 328386, 330797, 332165, 340684, 351612,
    356164, 356765, 370777, 377134, 379400, 381514, 388571, 394672, 396358,
    400995, 401673, 406460, 413189, 413513, 420020, 428615, 435203, 448412,
    454477, 456038, 460651, 463174, 464785, 465431, 469268,
]

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "zenodo"


def load_neural_dataset(subject_id: str = "all_hand_dmfc",
                        timebinsize: int = 50,
                        data_dir: Optional[Path] = None) -> dict:
    """
    Load the binned neural dataset.

    Args:
        subject_id: one of 'all_hand_dmfc', 'perle_hand_dmfc', 'mahler_hand_dmfc'
        timebinsize: bin size in ms (default 50)
        data_dir: path to data directory

    Returns:
        dict with keys like 'neural_responses', 'behavioral_responses', etc.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)

    fn = data_dir / f"{subject_id}_dataset_{timebinsize}ms.pkl"
    if not fn.exists():
        raise FileNotFoundError(
            f"Dataset not found at {fn}. "
            f"Download from https://zenodo.org/records/13952210"
        )
    with open(fn, "rb") as f:
        return pk.load(f)


def extract_initial_conditions(dat: dict) -> Dict[str, np.ndarray]:
    """
    Extract the 79 initial conditions (ball position and velocity) from the dataset.

    Args:
        dat: loaded dataset dict from load_neural_dataset()

    Returns:
        dict with keys:
            'ball_pos_x': (79, T) ball x positions over time
            'ball_pos_y': (79, T) ball y positions over time
            'ball_vel_x': (79, T) ball x velocities over time
            'ball_vel_y': (79, T) ball y velocities over time
            'ball_x0': (79,) initial x positions
            'ball_y0': (79,) initial y positions
            'vel_x0': (79,) initial x velocities
            'vel_y0': (79,) initial y velocities
    """
    beh = dat["behavioral_responses"]

    # Extract from visible epoch
    ball_pos_x = beh["vis"]["ball_pos_x"]  # (79, T)
    ball_pos_y = beh["vis"]["ball_pos_y"]  # (79, T)

    # Get initial velocities from metadata (heading + speed)
    meta_df = dat["meta"]
    heading_rad = np.radians(meta_df["py_heading0"].values)
    speed = meta_df["py_speed"].values
    vel_x0 = speed * np.cos(heading_rad)
    vel_y0 = speed * np.sin(heading_rad)

    # Initial positions: first non-NaN value per condition
    n_conds = ball_pos_x.shape[0]
    ball_x0 = np.zeros(n_conds)
    ball_y0 = np.zeros(n_conds)

    for i in range(n_conds):
        valid = np.isfinite(ball_pos_x[i])
        if valid.any():
            t0 = np.argmax(valid)
            ball_x0[i] = ball_pos_x[i, t0]
            ball_y0[i] = ball_pos_y[i, t0]

    return {
        "ball_pos_x": ball_pos_x,
        "ball_pos_y": ball_pos_y,
        "ball_x0": ball_x0,
        "ball_y0": ball_y0,
        "vel_x0": vel_x0,
        "vel_y0": vel_y0,
    }


def extract_neural_data(dat: dict, epoch: str = "occ") -> Dict[str, np.ndarray]:
    """
    Extract neural population activity.

    Args:
        dat: loaded dataset dict
        epoch: 'occ' for occluded epoch, 'vis' for visible epoch

    Returns:
        dict with keys:
            'firing_rates': (N_neurons, 79, T) trial-averaged firing rates
            't_from_start': (79, T) time from trial start in ms
    """
    # Use reliable neurons (1889) for occluded epoch if available
    if epoch == "occ" and "neural_responses_reliable" in dat:
        neural = dat["neural_responses_reliable"][epoch]
    else:
        neural = dat["neural_responses"][epoch]

    # Data is stored directly as ndarray, not as a dict
    firing_rates = neural if isinstance(neural, np.ndarray) else neural["data"]

    beh_epoch = dat["behavioral_responses"][epoch]
    t_from_start = beh_epoch.get("t_from_start", None) if isinstance(beh_epoch, dict) else None

    # Trim leading NaN bins (pre-trial padding, typically 6 bins)
    first_valid = 0
    for t in range(firing_rates.shape[2]):
        if not np.all(np.isnan(firing_rates[:, :, t])):
            first_valid = t
            break
    if first_valid > 0:
        firing_rates = firing_rates[:, :, first_valid:]

    return {
        "firing_rates": firing_rates,  # (N_neurons, 79, T)
        "t_from_start": t_from_start,
        "trimmed_bins": first_valid,
    }


def get_condition_metadata(dat: dict) -> Dict[str, np.ndarray]:
    """
    Extract per-condition metadata: which conditions bounce, endpoint, etc.

    Returns:
        dict with:
            'n_bounces': (79,) number of wall bounces per condition
            'ball_endpoint_y': (79,) y-position where ball reaches right edge
            'trial_duration': (79,) duration in time bins
    """
    beh = dat["behavioral_responses"]
    occ = beh["occ"]

    ball_pos_y = occ["ball_pos_y"]  # (79, T)

    # Endpoint y: use ball_final_y (constant per condition)
    endpoint_y = occ["ball_final_y"][:, 0]  # (79,) take first column

    # Trial duration: last valid time bin + 1
    n_conds = ball_pos_y.shape[0]
    trial_duration = np.zeros(n_conds, dtype=int)
    for i in range(n_conds):
        valid = np.isfinite(ball_pos_y[i])
        if valid.any():
            trial_duration[i] = np.max(np.where(valid)[0]) + 1

    # Get bounce count from the metadata DataFrame
    meta_df = dat["meta"]
    n_bounces = meta_df["py_n_bounce"].values.astype(int)

    return {
        "n_bounces": n_bounces,
        "ball_endpoint_y": endpoint_y,
        "trial_duration": trial_duration,
    }
