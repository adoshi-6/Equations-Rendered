import numpy as np
from PIL import Image, ImageDraw
import math
import os

from drawing_utils import draw_hatching, draw_spring, draw_mass

COLOR_BG = (0, 0, 0)
COLOR_AXIS = (100, 100, 100)
COLOR_VECTOR = (127, 174, 107) # Auxiliary
COLOR_RE = "#E85D4A" # Primary
COLOR_IM = "#5DA8E8" # Secondary

def recommended_duration(config: dict) -> float:
    return 12.0 # Fixed duration for Euler's formula demonstration

def generate(config: dict):
    duration = float(config.get("duration", 12.0))
    fps = int(config.get("fps", 30))
    num_frames = int(duration * fps)
    
    # 1. Simulate data (Euler's formula: e^{it} = cos(t) + i*sin(t))
    # Let's map t from 0 to 4*pi over the duration
    t_max = 4 * math.pi
    t_array = np.linspace(0, t_max, num_frames)
    
    re_array = np.cos(t_array)
    im_array = np.sin(t_array)
    
    # 2. Render frames
    w, h = 1080, 1080

    # PRECOMPUTE BOUNDS (Section 3.6)
    # Trajectory is unit circle: x in [-1, 1], y in [-1, 1].
    # Annotations: axes go from -1.333 to +1.333.
    # Hatching is at x = -1.333.
    min_x, max_x = -1.333, 1.333
    min_y, max_y = -1.333, 1.333
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    margin_x = range_x * 0.10
    margin_y = range_y * 0.10
    
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    
    scale_x = 1080 / (max_x - min_x)
    scale_y = 1080 / (max_y - min_y)
    radius = float(min(scale_x, scale_y))
    
    # In Euler, cx and cy are hardcoded. We compute them dynamically:
    cx = int(1080 / 2 - ((min_x + max_x) / 2) * radius)
    cy = int(1080 / 2 + ((min_y + max_y) / 2) * radius)

    
    # Pre-draw static background
    bg = Image.new("RGB", (w, h), COLOR_BG)
    draw_bg = ImageDraw.Draw(bg)
    
    # Draw axes
    draw_bg.line([(cx - int(1.333 * radius), cy), (cx + int(1.333 * radius), cy)], fill=COLOR_AXIS, width=2)
    draw_bg.line([(cx, cy - int(1.333 * radius)), (cx, cy + int(1.333 * radius))], fill=COLOR_AXIS, width=2)
    
    # Draw unit circle
    draw_bg.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=COLOR_AXIS, width=2)
    
    # Example usage of drawing primitives (just for POC)
    # We'll put a fixed wall at the left
    draw_hatching(draw_bg, (cx - int(1.333 * radius), cy - 100), (cx - int(1.333 * radius), cy + 100), normal_dir=(1, 0), color=COLOR_AXIS)
    
    logs = []

    def frame_generator():
        for i in range(num_frames):
            t = t_array[i]
            re, im = re_array[i], im_array[i]
            frame_img = bg.copy()
            draw = ImageDraw.Draw(frame_img)
        
            # Current position
            px = cx + int(re_array[i] * radius)
            py = cy - int(im_array[i] * radius) # -y is up in PIL
        
            # Draw spring from left wall to the mass
            draw_spring(draw, (cx - int(1.333 * radius), cy), (px, py), num_coils=8, color=(150, 150, 150))
        
            # Draw vector
            draw.line([(cx, cy), (px, py)], fill=COLOR_VECTOR, width=4)
        
            # Draw mass box at the end
            draw_mass(draw, (px, py), (40, 40), color=COLOR_VECTOR, outline=None)
        
            # Draw projections
            draw.line([(px, py), (px, cy)], fill=COLOR_IM, width=2) # vertical (Im)
            draw.line([(px, py), (cx, py)], fill=COLOR_RE, width=2) # horizontal (Re)

            # Log variables incrementally, in lockstep with frame generation.
            # CRITICAL: this MUST append here (before yield), not in a separate
            # loop after frame_generator() is defined. renderer.py reads
            # variable_logs[-1] on every composited frame; if this list were
            # fully pre-built before any frame is yielded, every single frame
            # of the rendered video would show the same (final) Angle/Re/Im
            # values, frozen for the whole render. (This was a real, confirmed
            # bug — fixed here.)
            logs.append([
                {"name": "Angle", "value": f"{t:.2f} rad", "role": "metric", "metric_index": 0},
                {"name": "Re", "value": f"{re:.2f}", "role": "primary"},
                {"name": "Im", "value": f"{im:.2f}", "role": "secondary"}
            ])
        
            yield np.array(frame_img)
        
    auxiliary_curves = {
        "time": t_array,
        "series": {
            "cos(t)": {"data": re_array, "color": COLOR_RE},
            "sin(t)": {"data": im_array, "color": COLOR_IM}
        },
        "xlabel": "t (radians)",
        "ylabel": "Amplitude"
    }
        
    return frame_generator(), logs, auxiliary_curves
