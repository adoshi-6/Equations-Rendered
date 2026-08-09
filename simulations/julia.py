import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

COLOR_ROLE_1 = xp.array([232.0, 93.0, 74.0])     # Primary (#E85D4A)
COLOR_ROLE_2 = xp.array([93.0, 168.0, 232.0])    # Secondary (#5DA8E8)
COLOR_ROLE_3 = xp.array([127.0, 174.0, 107.0])   # Auxiliary (#7FAE6B)
COLOR_TRAIL = xp.array([93.0, 168.0, 232.0])     # Match Secondary

def recommended_duration(config: dict) -> float:
    return 20.0

def get_palette_color(t, active):
    t_3d = t[:, :, None]
    
    w1 = (xp.sin(2 * xp.pi * t_3d) + 1) / 2
    w2 = (xp.sin(2 * xp.pi * t_3d + 2 * xp.pi / 3) + 1) / 2
    w3 = (xp.sin(2 * xp.pi * t_3d + 4 * xp.pi / 3) + 1) / 2
    
    c1 = COLOR_ROLE_2[None, None, :]
    c2 = COLOR_ROLE_3[None, None, :]
    c3 = COLOR_TRAIL[None, None, :]
    
    color = (w1 * c1 + w2 * c2 + w3 * c3) / (w1 + w2 + w3)
    color = xp.clip(color, 0, 255)
    color[active] = 0
    
    return color.astype(xp.uint8)

def evaluate_point(inp):
    x, y = inp
    c = -0.8 + 0.156j
    z = x + 1j * y
    escapes = False
    for _ in range(120):
        z = z**2 + c
        if (z.real**2 + z.imag**2) > 4.0:
            escapes = True
            break
    return {"escapes": escapes}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        ((-0.5275031186435346, 0.07591217835228786), {"escapes": False}, None),
        ((2.0, 2.0), {"escapes": True}, None)
    ]
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    duration = config.get("duration", 20.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    
    # Target feature near boundary
    x_center = -0.8
    y_center = 0.156
    
    w_start = 3.0
    w_end = 0.005
    
    zoom_rate = (w_end / w_start) ** (1.0 / num_frames)
    max_iter = 120
    variable_logs = []
    
    u = xp.linspace(-0.5, 0.5, width)
    v = xp.linspace(-0.5, 0.5, height)
    uu, vv = xp.meshgrid(u, v)
    
    print("Generating Julia Set frames...")
    
    # Julia constant
    C = -0.8 + 0.156j
    
    def frame_generator():
        for f in range(num_frames):
            w = w_start * (zoom_rate ** f)
        
            c_real = x_center + uu * w
            c_imag = y_center + vv * w
            z = c_real + 1j * c_imag
        
            iterations = xp.zeros(z.shape, dtype=xp.float32)
            active = xp.ones(z.shape, dtype=xp.bool_)
        
            for n in range(max_iter):
                z[active] = z[active]**2 + C
            
                escaped = (z.real**2 + z.imag**2) > 4.0
                newly_escaped = escaped & active
            
                iterations[newly_escaped] = n
                active[newly_escaped] = False
            
                if not xp.any(active):
                    break
                
            zoom_level = 3.0 / w
            variable_logs.append([
                {"name": "Zoom Depth", "value": f"{zoom_level:.1f}x", "role": "metric", "metric_index": 0},
                {"name": "Max Iterations", "value": str(max_iter), "role": "metric", "metric_index": 1}
            ])
        
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
            
            yield frame
        
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frame_generator(), variable_logs, None
