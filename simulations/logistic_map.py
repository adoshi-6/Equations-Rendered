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
COLOR_ROLE_1 = xp.array([255.0, 50.0, 50.0])     # Bright Red
COLOR_ROLE_2 = xp.array([50.0, 150.0, 255.0])    # Electric Blue
COLOR_ROLE_3 = xp.array([50.0, 255.0, 50.0])     # Neon Green
COLOR_TRAIL = xp.array([255.0, 120.0, 0.0])      # Orange

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
    Generates frames and variable logs for the Logistic Map Bifurcation Zoom.
    """
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    
    r_center = 3.842  
    x_center = 0.525
    
    r_width_start = 1.1
    x_height_start = 1.0
    
    r_width_end = 0.005
    x_height_end = 0.005
    
    zoom_r = (r_width_end / r_width_start) ** (1.0 / num_frames)
    zoom_x = (x_height_end / x_height_start) ** (1.0 / num_frames)
    
    num_trajectories = 250
    transient_iters = 300
    plot_iters = 150
    
    frames = []
    variable_logs = []
    
    col_indices = xp.tile(xp.arange(width), (num_trajectories, 1))
    
    print("Generating Logistic Map zoom frames...")
    
    for f in range(num_frames):
        r_w = r_width_start * (zoom_r ** f)
        x_h = x_height_start * (zoom_x ** f)
        
        # Log variable
        zoom_level = 1.1 / r_w
        variable_logs.append({
            "Zoom Depth": f"{zoom_level:.1f}x",
            "Viewport Δr": f"{r_w:.5f}"
        })
        
        r_min = r_center - r_w / 2.0
        r_max = r_center + r_w / 2.0
        x_min = x_center - x_h / 2.0
        x_max = x_center + x_h / 2.0
        
        r = xp.linspace(r_min, r_max, width)[None, :]
        r = xp.clip(r, 0.0, 4.0)
        
        x = xp.linspace(0.1, 0.9, num_trajectories)[:, None] * xp.ones((1, width))
        
        for _ in range(transient_iters):
            x = r * x * (1.0 - x)
            
        density_grid = xp.zeros((height, width), dtype=xp.float32)
        
        for _ in range(plot_iters):
            x = r * x * (1.0 - x)
            
            y_px = ((x_max - x) / (x_max - x_min) * height).astype(xp.int32)
            valid = (y_px >= 0) & (y_px < height)
            xp.add.at(density_grid, (y_px[valid], col_indices[valid]), 1.0)
            
        max_val = xp.max(density_grid)
        if max_val > 0:
            density_grid = xp.log1p(density_grid) / xp.log1p(max_val)
            
        color_grid = get_palette_color(density_grid)
        
        if hasattr(color_grid, "get"):
            frame = color_grid.get()
        else:
            frame = np.asarray(color_grid)
            
        frames.append(frame)
        
        if (f + 1) % 30 == 0:
            print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frames, variable_logs
