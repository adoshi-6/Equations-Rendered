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

# Desaturated shared palette
ROLE_COLORS = {
    "primary": xp.array([232.0, 93.0, 74.0]),
    "secondary": xp.array([93.0, 168.0, 232.0]),
    "auxiliary": xp.array([127.0, 174.0, 107.0]),
    "control": xp.array([212.0, 194.0, 74.0]),
    "static": xp.array([168.0, 181.0, 194.0]),
}
METRIC_COLORS_RGB = [
    (232, 144, 93),
    (184, 127, 201),
    (201, 127, 160),
    (127, 201, 176),
]
# Base colors for the fractal's own cyclic iteration-count palette — this is
# a distinct concept from ROLE_COLORS (no single "role" maps to a fractal's
# escape-time coloring), so it keeps its own cyclic blend, just shifted into
# the same desaturated hue family instead of pure neon.
FRACTAL_COLOR_1 = xp.array([93.0, 168.0, 232.0])   # secondary blue
FRACTAL_COLOR_2 = xp.array([127.0, 174.0, 107.0])  # auxiliary green
FRACTAL_COLOR_3 = xp.array([232.0, 144.0, 93.0])   # metric0 amber

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
    c1 = FRACTAL_COLOR_1[None, None, :]
    c2 = FRACTAL_COLOR_2[None, None, :]
    c3 = FRACTAL_COLOR_3[None, None, :]
    
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
    variable_logs = []
    
    u = xp.linspace(-0.5, 0.5, width)
    v = xp.linspace(-0.5, 0.5, height)
    uu, vv = xp.meshgrid(u, v)
    
    print("Generating Mandelbrot zoom frames...")
    
    def frame_generator():
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
                
            # Calculate current zoom level for logs.
            # Zoom Depth is a genuine aggregate (no single corresponding
            # element) -> metric. Max Iterations is a fixed constant that
            # never changes across the render -> static, not metric
            # (reclassified from an earlier miscategorization — a value
            # that never varies shouldn't share the "metric" role meant for
            # dynamic aggregate readouts).
            zoom_level = 3.0 / w
            variable_logs.append([
                {"name": "Zoom Depth", "value": f"{zoom_level:.1f}x", "role": "metric", "metric_index": 0},
                {"name": "Max Iterations", "value": str(max_iter), "role": "static"},
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
