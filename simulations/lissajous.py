import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

# Universal Palette
COLOR_ROLE_1 = (232, 93, 74)     # Primary (#E85D4A)
COLOR_ROLE_2 = (93, 168, 232)    # Secondary (#5DA8E8)
COLOR_ROLE_3 = (127, 174, 107)   # Auxiliary (#7FAE6B)
COLOR_TRAIL = (93, 168, 232)     # Match Secondary

def evaluate_point(inp):
    t, delta = inp["t"], inp["delta"]
    fx, fy = inp.get("freq_x", 3.0), inp.get("freq_y", 4.0)
    x = float(np.sin(fx * t + delta))
    y = float(np.sin(fy * t))
    return {"x": x, "y": y}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        ({"t": 0.0, "delta": 0.0}, {"x": 0.0, "y": 0.0}, 1e-6),
        ({"t": np.pi/2, "delta": 0.0, "freq_x": 1.0, "freq_y": 1.0}, {"x": 1.0, "y": 1.0}, 1e-6),
    ],
}

def recommended_duration(config: dict) -> float:
    """
    Computes duration for a full 2*pi phase sweep of the Lissajous curve.
    10 seconds provides a smooth and complete sweep.
    """
    return 10.0

def get_gradient_color(i, n):
    """
    Generates a gradient using the universal palette (Blue -> Green).
    """
    factor = i / max(1, n - 1)
    
    r = int(COLOR_ROLE_2[0] * (1.0 - factor) + COLOR_ROLE_3[0] * factor)
    g = int(COLOR_ROLE_2[1] * (1.0 - factor) + COLOR_ROLE_3[1] * factor)
    b = int(COLOR_ROLE_2[2] * (1.0 - factor) + COLOR_ROLE_3[2] * factor)
    
    return (r, g, b)

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Lissajous curves simulation.
    """
    duration = config.get("duration", 10.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    center_x, center_y = width / 2, height / 2
    
    # PRECOMPUTE BOUNDS (Section 3.6)
    # The mathematical extent of sin() is [-1, 1]. Adding 10% margin gives [-1.1, 1.1]
    range_span = 2.2
    scale = float(min(width / range_span, height / range_span))
    
    freq_x = 3.0
    freq_y = 4.0
    
    num_curves = 100
    num_points = 600
    t = xp.linspace(0, 2.0 * xp.pi, num_points)
    variable_logs = []
    
    print("Generating Lissajous curve frames...")
    
    def frame_generator():
        for f in range(num_frames):
            base_delta = 2.0 * xp.pi * (f / max(1, num_frames - 1))
        
            # Log variable (Section 3.3)
            # 0: amber, 1: purple
            variable_logs.append([
                {"name": "Ratio", "value": f"{freq_x:.0f}:{freq_y:.0f}", "role": "metric", "metric_index": 1},
                {"name": "Phase (δ)", "value": f"{base_delta:.2f} rad", "role": "metric", "metric_index": 0}
            ])
        
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)
        
            delta = base_delta + xp.linspace(-0.25, 0.25, num_curves)[:, None]
        
            x_coords = xp.sin(freq_x * t[None, :] + delta)
            y_coords = xp.sin(freq_y * t[None, :]) * xp.ones((num_curves, 1))
        
            px = (center_x + x_coords * scale)
            py = (center_y - y_coords * scale)
        
            if hasattr(px, "get"):
                px_cpu = px.get()
                py_cpu = py.get()
            else:
                px_cpu = np.asarray(px)
                py_cpu = np.asarray(py)
            
            for i in range(num_curves):
                # Draw the trail using a single primary color with a soft fading alpha (max 40)
                alpha = int(40 * (i / max(1, num_curves - 1)))
                color = tuple(int(c) for c in COLOR_ROLE_1)
            
                pts = [(px_cpu[i, k], py_cpu[i, k]) for k in range(num_points)]
                draw.line(pts, fill=color + (alpha,), width=2)
            
            yield np.array(img.convert("RGB"))
        
    return frame_generator(), variable_logs, None
