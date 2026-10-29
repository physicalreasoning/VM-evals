"""
Pong stimulus renderer for the 79 MentalPong conditions.

Renders the ball interception task as video frames.
The ball travels rightward at constant speed with zero or one bounce off
horizontal walls. The ball is occluded partway through the trial.

Two rendering modes:
  1. From dataset TRUE positions — uses the exact ball trajectory from the data
  2. From physics simulation — simulates from initial conditions
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple


# Board coordinate ranges (from dataset)
BOARD_X_MIN = -10.0
BOARD_X_MAX = 11.0
BOARD_Y_MIN = -10.0
BOARD_Y_MAX = 10.0

# Rendering defaults
DEFAULT_FRAME_SIZE = (224, 224)  # H, W — matches common model input size
BALL_RADIUS_PX = 5
PADDLE_WIDTH_PX = 4
PADDLE_HEIGHT_PX = 20
OCCLUDER_COLOR = (128, 0, 0)     # dark red, matching the task
BALL_COLOR = (255, 255, 255)
PADDLE_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
WALL_COLOR = (100, 100, 100)

FPS = 20  # 50ms bins → 20 fps


class PongRenderer:
    """Renders Pong ball positions as video frames."""

    def __init__(self, frame_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
                 ball_radius: int = BALL_RADIUS_PX,
                 x_range: Tuple[float, float] = (BOARD_X_MIN, BOARD_X_MAX),
                 y_range: Tuple[float, float] = (BOARD_Y_MIN, BOARD_Y_MAX)):
        self.frame_h, self.frame_w = frame_size
        self.ball_radius = ball_radius
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def _pos_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert board coordinates to pixel coordinates."""
        px = int((x - self.x_min) / (self.x_max - self.x_min) * self.frame_w)
        py = int((1.0 - (y - self.y_min) / (self.y_max - self.y_min)) * self.frame_h)
        return px, py

    def render_frame(self, ball_x: float, ball_y: float,
                     occluder_x: Optional[float] = None,
                     show_ball: bool = True,
                     paddle_y: Optional[float] = None) -> np.ndarray:
        """
        Render a single frame.

        Args:
            ball_x, ball_y: ball position in board coordinates
            occluder_x: x-coordinate where occluder starts (None = no occluder)
            show_ball: whether to draw the ball
            paddle_y: paddle y-position in board coordinates (optional)

        Returns:
            frame: (H, W, 3) uint8 array
        """
        frame = np.full((self.frame_h, self.frame_w, 3), BG_COLOR, dtype=np.uint8)

        # Draw walls (top and bottom borders)
        frame[0:2, :] = WALL_COLOR
        frame[-2:, :] = WALL_COLOR

        # Draw occluder
        if occluder_x is not None:
            occ_px, _ = self._pos_to_pixel(occluder_x, 0)
            occ_px = max(0, min(occ_px, self.frame_w))
            frame[:, occ_px:] = np.array(OCCLUDER_COLOR, dtype=np.uint8)

        # Draw ball
        if show_ball:
            bx, by = self._pos_to_pixel(ball_x, ball_y)
            r = self.ball_radius
            y_min = max(0, by - r)
            y_max = min(self.frame_h, by + r + 1)
            x_min = max(0, bx - r)
            x_max = min(self.frame_w, bx + r + 1)

            for iy in range(y_min, y_max):
                for ix in range(x_min, x_max):
                    if (ix - bx) ** 2 + (iy - by) ** 2 <= r ** 2:
                        frame[iy, ix] = BALL_COLOR

        # Draw paddle
        if paddle_y is not None:
            px = self.frame_w - PADDLE_WIDTH_PX
            _, py_center = self._pos_to_pixel(self.x_max, paddle_y)
            ph = PADDLE_HEIGHT_PX // 2
            y_min = max(0, py_center - ph)
            y_max = min(self.frame_h, py_center + ph + 1)
            frame[y_min:y_max, px:] = PADDLE_COLOR

        return frame

    def render_trial(self, ball_x: np.ndarray, ball_y: np.ndarray,
                     occluder_x: Optional[float] = None) -> np.ndarray:
        """
        Render a full trial as a sequence of frames.

        Args:
            ball_x: (T,) x-positions at each time bin
            ball_y: (T,) y-positions at each time bin
            occluder_x: x-coordinate where occluder starts

        Returns:
            frames: (T_valid, H, W, 3) uint8 array (skips NaN bins)
        """
        frames = []
        for t in range(len(ball_x)):
            bx, by = ball_x[t], ball_y[t]
            if np.isnan(bx) or np.isnan(by):
                continue

            show_ball = True
            if occluder_x is not None and bx >= occluder_x:
                show_ball = False

            frame = self.render_frame(bx, by, occluder_x=occluder_x,
                                      show_ball=show_ball)
            frames.append(frame)

        return np.array(frames) if frames else np.zeros((0, self.frame_h, self.frame_w, 3), dtype=np.uint8)


def render_all_conditions(dat: dict,
                          frame_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
                          include_occluded: bool = True,
                          visible_only: bool = False) -> dict:
    """
    Render all 79 conditions using TRUE positions from the dataset.

    Args:
        dat: loaded dataset dict from load_neural_dataset()
        frame_size: (H, W) pixel size for rendered frames
        include_occluded: if True, render full trial (vis + occ);
                          if False, only render visible epoch
        visible_only: if True, ball is hidden once occluded (more realistic);
                      if False, ball is always visible (for model comparison)

    Returns:
        dict with:
            'frames': list of 79 np.ndarray, each (T_i, H, W, 3)
            'n_frames': (79,) number of frames per condition
            'ball_x': list of 79 np.ndarray, each (T_i,)
            'ball_y': list of 79 np.ndarray, each (T_i,)
    """
    beh = dat["behavioral_responses"]

    vis_x = beh["vis"]["ball_pos_x_TRUE"]  # (79, T_vis)
    vis_y = beh["vis"]["ball_pos_y_TRUE"]  # (79, T_vis)
    occ_x = beh["occ"]["ball_pos_x_TRUE"]  # (79, T_occ)
    occ_y = beh["occ"]["ball_pos_y_TRUE"]  # (79, T_occ)

    # Compute occluder x for each condition from occ_alpha
    meta = dat["meta"]
    occ_alpha = meta["occ_alpha"].values  # fraction of board traversed at occlusion

    renderer = PongRenderer(frame_size=frame_size)

    all_frames = []
    all_ball_x = []
    all_ball_y = []
    n_frames = np.zeros(79, dtype=int)

    for i in range(79):
        # Concatenate visible + occluded trajectories
        if include_occluded:
            bx = np.concatenate([vis_x[i], occ_x[i]])
            by = np.concatenate([vis_y[i], occ_y[i]])
        else:
            bx = vis_x[i].copy()
            by = vis_y[i].copy()

        # Remove NaN padding
        valid = np.isfinite(bx) & np.isfinite(by)
        bx_valid = bx[valid]
        by_valid = by[valid]

        # Occluder position: the x value where the ball crosses into occlusion
        # Use the first occ position as approximation
        occ_valid = np.isfinite(occ_x[i])
        if occ_valid.any():
            occluder_x_val = occ_x[i][np.argmax(occ_valid)]
        else:
            occluder_x_val = None

        if visible_only:
            frames = renderer.render_trial(bx_valid, by_valid,
                                           occluder_x=occluder_x_val)
        else:
            # Ball always visible (no occluder drawn)
            frames = renderer.render_trial(bx_valid, by_valid,
                                           occluder_x=None)

        all_frames.append(frames)
        all_ball_x.append(bx_valid)
        all_ball_y.append(by_valid)
        n_frames[i] = len(frames)

    return {
        "frames": all_frames,
        "n_frames": n_frames,
        "ball_x": all_ball_x,
        "ball_y": all_ball_y,
    }


def save_frames_as_images(frames: np.ndarray, output_dir: str,
                          prefix: str = "frame"):
    """Save frames as individual PNG images."""
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"{prefix}_{i:04d}.png"))


def save_frames_as_video(frames: np.ndarray, output_path: str,
                         fps: int = FPS):
    """Save frames as an MP4 video."""
    import cv2
    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
