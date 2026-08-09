import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

# Universal Palette
COLOR_ROLE_1 = xp.array([232.0, 93.0, 74.0])     # Primary (#E85D4A)
COLOR_ROLE_2 = xp.array([93.0, 168.0, 232.0])    # Secondary (#5DA8E8)
COLOR_ROLE_3 = xp.array([127.0, 174.0, 107.0])   # Auxiliary (#7FAE6B)
COLOR_TRAIL = xp.array([93.0, 168.0, 232.0])     # Match Secondary

def recommended_duration(config: dict) -> float:
    """
    Computes duration for a parameter sweep of the Hénon Map.
    A 12-second duration provides a good visualization pace.
    """
    return 12.0

def get_gradient_color(density, active_pixels):
    """
    Maps density to a glowing universal palette gradient.
    """
    img = xp.zeros((1080, 1080, 3), dtype=xp.float32)
    
    d = density[:, :, None]
    
    # We map Electric Blue (low density) to Bright Red (high density)
    color_low = COLOR_ROLE_2
    color_high = COLOR_ROLE_1
    
    img[active_pixels] = (d[active_pixels] * color_high + (1.0 - d[active_pixels]) * color_low)
    
    brightness = xp.clip(density * 1.5, 0, 1)[:, :, None]
    img = img * brightness
    
    return xp.clip(img, 0, 255).astype(xp.uint8)

def simulate_headless(config: dict):
    duration = config.get("duration", 2.0)
    
    a, b = 1.4, 0.3
    num_points = 500
    
    # Initialize points at origin (known to be in the basin of attraction)
    x = xp.zeros(num_points)
    y = xp.zeros(num_points)
    
    num_frames = int(duration * 30) # mock fps
    iters_per_frame = 5
    
    states = []
    
    for f in range(num_frames):
        for _ in range(iters_per_frame):
            x_new = 1.0 - a * x**2 + y
            y_new = b * x
            x, y = x_new, y_new
    
        xc = x.get() if hasattr(x, "get") else np.asarray(x)
        yc = y.get() if hasattr(y, "get") else np.asarray(y)
        states.append(np.stack([xc, yc], axis=-1))
        
    return np.arange(num_frames), states

def get_state_variables(states):
    stacked = np.stack(states, axis=0) # shape (T, N, 2)
    return {
        "x": stacked[:, :, 0],
        "y": stacked[:, :, 1]
    }

TEST_SPEC = {
    "category": "bounded_region",
    "expected_bounds": {
        "x": {"min": -2.5, "max": 2.5},
        "y": {"min": -1.5, "max": 1.5}
    }
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Hénon Map parameter sweep.
    """
    duration = config.get("duration", 12.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    
    a_start = 1.0
    a_end = 1.4
    b = 0.3
    
    num_trajectories = 1200
    warmup_iters = 180
    plot_iters = 100
    variable_logs = []
    
    # PRECOMPUTE BOUNDS (Section 3.6)
    print("Precomputing physical bounds...")
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    # Sample parameter 'a' across the sweep
    a_samples = np.linspace(a_start, a_end, 30)
    for a_val in a_samples:
        xp.random.seed(42)
        x_st = xp.random.uniform(-0.5, 0.5, (num_trajectories,))
        y_st = xp.random.uniform(-0.1, 0.1, (num_trajectories,))
        
        for _ in range(warmup_iters):
            x_next = 1.0 - a_val * x_st**2 + y_st
            y_st = b * x_st
            x_st = x_next
            
        for _ in range(plot_iters):
            x_next = 1.0 - a_val * x_st**2 + y_st
            y_st = b * x_st
            x_st = x_next
            min_x = min(min_x, float(xp.min(x_st)))
            max_x = max(max_x, float(xp.max(x_st)))
            min_y = min(min_y, float(xp.min(y_st)))
            max_y = max(max_y, float(xp.max(y_st)))
            
    # Add 10% margin
    range_x = max(max_x - min_x, 1e-3)
    range_y = max(max_y - min_y, 1e-3)
    margin_x = range_x * 0.10
    margin_y = range_y * 0.10
    x_min = min_x - margin_x
    x_max = max_x + margin_x
    y_min = min_y - margin_y
    y_max = max_y + margin_y
    
    def frame_generator():
        print("Generating Hénon Map sweep frames...")
        
        for f in range(num_frames):
            nonlocal a_start, a_end, b
            
            t = f / max(1, num_frames - 1)
            a = a_start + t * (a_end - a_start)
            
            # Log variables (Section 3.3)
            variable_logs.append([
                {"name": "Parameter a", "value": f"{a:.3f}", "role": "metric", "metric_index": 0},
                {"name": "Parameter b", "value": f"{b:.1f}", "role": "metric", "metric_index": 1}
            ])
            
            xp.random.seed(42)
            x_st = xp.random.uniform(-0.5, 0.5, (num_trajectories,))
            y_st = xp.random.uniform(-0.1, 0.1, (num_trajectories,))
            
            for _ in range(warmup_iters):
                x_next = 1.0 - a * x_st**2 + y_st
                y_st = b * x_st
                x_st = x_next
                
            xs = xp.zeros((plot_iters, num_trajectories))
            ys = xp.zeros((plot_iters, num_trajectories))
            
            for k in range(plot_iters):
                x_next = 1.0 - a * x_st**2 + y_st
                y_st = b * x_st
                x_st = x_next
                xs[k] = x_st
                ys[k] = y_st
                
            all_x = xs.flatten()
            all_y = ys.flatten()
            
            grid = xp.zeros((height, width), dtype=xp.float32)
            
            px = ((all_x - x_min) / (x_max - x_min) * width).astype(xp.int32)
            py = ((y_max - all_y) / (y_max - y_min) * height).astype(xp.int32) 
            
            valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            
            xp.add.at(grid, (py[valid], px[valid]), 1.0)
            
            max_val = xp.max(grid)
            if max_val > 0:
                grid = xp.log1p(grid) / xp.log1p(max_val)
                
            active_pixels = grid > 0.005
            
            color_grid = get_gradient_color(grid, active_pixels)
            
            if hasattr(color_grid, "get"):
                frame = color_grid.get()
            else:
                frame = np.asarray(color_grid)
                
            yield frame
            
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frame_generator(), variable_logs, None
