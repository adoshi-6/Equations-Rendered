import os
import sys
import numpy as np

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

def evaluate_point(inp):
    c_val = inp["c"]
    c = complex(c_val, 0)
    z = 0j
    for _ in range(100):
        if abs(z) > 2.0:
            return {"in_set": False}
        z = z*z + c
    return {"in_set": True}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        ({"c": -1.0}, {"in_set": True}, None),
        ({"c": 1.0}, {"in_set": False}, None),
    ],
}

# Universal Palette
COLOR_ROLE_1 = xp.array([255.0, 50.0, 50.0])     # Bright Red
COLOR_ROLE_2 = xp.array([50.0, 150.0, 255.0])    # Electric Blue
COLOR_ROLE_3 = xp.array([50.0, 255.0, 50.0])     # Neon Green
COLOR_TRAIL = xp.array([255.0, 120.0, 0.0])      # Orange

def recommended_duration(config: dict) -> float:
    """
    Computes duration for zooming from w=3.0 down to w=0.00005.
    A steady, visually pleasing pace is a 10% zoom per second.
    At 10% per second (factor 0.9): 3.0 * (0.9)^t = 0.00005
    t = log(0.00005 / 3.0) / log(0.9) ≈ 104s.
    That's too long. Let's aim for a smooth zoom covering the distance in 20 seconds.
    """
    return 20.0

def get_palette_color(t, active):
    """
    Uses the universal palette to color the Mandelbrot set.
    We blend Blue -> Green -> Red -> Orange over the iteration range.
    """
    # Shift and scale t to cycle through the colors
    # Map t in [0, 1] to a cyclic cosine interpolation
    t_3d = t[:, :, None]
    
    # We construct a simple cyclic palette using sine/cosine and the universal colors
    # This creates bands of the universal colors.
    w1 = (xp.sin(2 * xp.pi * t_3d) + 1) / 2
    w2 = (xp.sin(2 * xp.pi * t_3d + 2 * xp.pi / 3) + 1) / 2
    w3 = (xp.sin(2 * xp.pi * t_3d + 4 * xp.pi / 3) + 1) / 2
    
    # Base colors (reshaped for broadcasting)
    c1 = COLOR_ROLE_2[None, None, :] # Blue
    c2 = COLOR_ROLE_3[None, None, :] # Green
    c3 = COLOR_TRAIL[None, None, :]  # Orange
    
    color = (w1 * c1 + w2 * c2 + w3 * c3) / (w1 + w2 + w3)
    
    color = xp.clip(color, 0, 255)
    color[active] = 0 # Inside set is black
    
    return color.astype(xp.uint8)

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Mandelbrot zoom simulation.
    """
    duration = config.get("duration", 20.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    
    x_center = -0.7436438870371587
    y_center = 0.13182590420531197
    
    w_start = 3.0
    w_end = 0.00005
    
    zoom_rate = (w_end / w_start) ** (1.0 / num_frames)
    max_iter = 120
    
    frames = []
    variable_logs = []
    
    u = xp.linspace(-0.5, 0.5, width)
    v = xp.linspace(-0.5, 0.5, height)
    uu, vv = xp.meshgrid(u, v)
    
    print("Generating Mandelbrot zoom frames...")
    
    for f in range(num_frames):
        w = w_start * (zoom_rate ** f)
        
        c_real = x_center + uu * w
        c_imag = y_center + vv * w
        c = c_real + 1j * c_imag
        
        z = xp.zeros_like(c)
        iterations = xp.zeros(c.shape, dtype=xp.float32)
        active = xp.ones(c.shape, dtype=xp.bool_)
        
        for n in range(max_iter):
            z[active] = z[active]**2 + c[active]
            
            escaped = (z.real**2 + z.imag**2) > 4.0
            newly_escaped = escaped & active
            
            iterations[newly_escaped] = n
            active[newly_escaped] = False
            
            if not xp.any(active):
                break
                
        # Calculate current zoom level for logs
        zoom_level = 3.0 / w
        variable_logs.append({
            "Zoom Depth": f"{zoom_level:.1f}x",
            "Max Iterations": str(max_iter)
        })
        
        escaped_mask = ~active
        if xp.any(escaped_mask):
            z_esc = z[escaped_mask]
            log_zn = xp.log(z_esc.real**2 + z_esc.imag**2) / 2.0
            nu = xp.log(log_zn / xp.log(2.0)) / xp.log(2.0)
            iterations[escaped_mask] = iterations[escaped_mask] + 1.0 - nu
            
        norm_iter = iterations / max_iter
        color_grid = get_palette_color(norm_iter, active)
        
        if hasattr(color_grid, "get"):
            frame = color_grid.get()
        else:
            frame = np.asarray(color_grid)
            
        frames.append(frame)
        
        if (f + 1) % 30 == 0:
            print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frames, variable_logs
