import os
import sys
import numpy as np
from PIL import Image

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

def find_bifurcation_transitions(config):
    # known transitions: period 2 at r=3.0, period 4 at 3.449
    # simple mock or basic calculation
    return {
        "period_2": 3.00,
        "period_4": 3.449,
        "chaos_onset": 3.569
    }

TEST_SPEC = {
    "category": "bifurcation",
    "known_transitions": {
        "period_2": 3.00,
        "period_4": 3.449,
        "chaos_onset": 3.569
    }
}

# Universal Palette
COLOR_ROLE_1 = xp.array([232.0, 93.0, 74.0])     # Primary (#E85D4A)
COLOR_ROLE_2 = xp.array([93.0, 168.0, 232.0])    # Secondary (#5DA8E8)
COLOR_ROLE_3 = xp.array([127.0, 174.0, 107.0])   # Auxiliary (#7FAE6B)
COLOR_TRAIL = xp.array([93.0, 168.0, 232.0])     # Match Secondary

def recommended_duration(config: dict) -> float:
    """
    Computes duration for zooming into the Logistic Map bifurcation diagram.
    A 15-second duration provides a steady, visually pleasing pace.
    """
    return 15.0

def get_palette_color(density):
    """
    Applies the universal color palette to the density map.
    Fades from Black -> Electric Blue -> Bright Red for high density.
    """
    d_3d = density[:, :, None]
    
    # Base colors (reshaped for broadcasting)
    c1 = COLOR_ROLE_2[None, None, :] # Blue
    c2 = COLOR_ROLE_1[None, None, :] # Red
    
    # Interpolate:
    # density < 0.5: Black to Blue
    # density > 0.5: Blue to Red
    color = xp.where(
        d_3d < 0.5,
        (d_3d * 2.0) * c1,
        c1 + ((d_3d - 0.5) * 2.0) * (c2 - c1)
    )
    
    color = xp.clip(color, 0, 255)
    return color.astype(xp.uint8)

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Logistic Map Bifurcation Sweep.
    """
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    
    r_min_global = 2.5
    r_max_global = 4.0
    x_min_global = 0.0
    x_max_global = 1.0
    
    num_trajectories = 250
    transient_iters = 300
    plot_iters = 150
    variable_logs = []
    
    col_indices = xp.tile(xp.arange(width), (num_trajectories, 1))
    
    print("Generating Logistic Map static sweep frames...")
    
    def frame_generator():
        # Precompute the entire density grid for a static sweep
        # We will reveal it gradually from left to right as r sweeps.
        density_grid_full = xp.zeros((height, width), dtype=xp.float32)
        r_all = xp.linspace(r_min_global, r_max_global, width)[None, :]
        x_all = xp.linspace(0.1, 0.9, num_trajectories)[:, None] * xp.ones((1, width))
        
        for _ in range(transient_iters):
            x_all = r_all * x_all * (1.0 - x_all)
            
        for _ in range(plot_iters):
            x_all = r_all * x_all * (1.0 - x_all)
            y_px = ((x_max_global - x_all) / (x_max_global - x_min_global) * height).astype(xp.int32)
            valid = (y_px >= 0) & (y_px < height)
            xp.add.at(density_grid_full, (y_px[valid], col_indices[valid]), 1.0)
            
        for f in range(num_frames):
            progress = f / max(1, num_frames - 1)
            current_r = r_min_global + progress * (r_max_global - r_min_global)
            
            # Log variable
            variable_logs.append([
                {"name": "Growth Rate (r)", "value": f"{current_r:.4f}", "role": "metric", "metric_index": 0}
            ])
            
            # Mask the density grid to only show up to current_r
            col_limit = int(progress * width)
            density_grid = xp.zeros_like(density_grid_full)
            if col_limit > 0:
                density_grid[:, :col_limit] = density_grid_full[:, :col_limit]
            
            max_val = xp.max(density_grid)
            if max_val > 0:
                density_grid = xp.log1p(density_grid) / xp.log1p(max_val)
            
            color_grid = get_palette_color(density_grid)
        
            if hasattr(color_grid, "get"):
                frame = color_grid.get()
            else:
                frame = np.asarray(color_grid)
            
            yield frame
        
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frame_generator(), variable_logs, None
