import numpy as np
import cv2
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
    variable_logs = []
    logs = []
    
    cx, cy = w // 2, h // 2
    radius = 300
    
    # Pre-draw static background
    bg = Image.new("RGB", (w, h), COLOR_BG)
    draw_bg = ImageDraw.Draw(bg)
    
    # Draw axes
    draw_bg.line([(cx - 400, cy), (cx + 400, cy)], fill=COLOR_AXIS, width=2)
    draw_bg.line([(cx, cy - 400), (cx, cy + 400)], fill=COLOR_AXIS, width=2)
    
    # Draw unit circle
    draw_bg.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=COLOR_AXIS, width=2)
    
    # Example usage of drawing primitives (just for POC)
    # We'll put a fixed wall at the left
    draw_hatching(draw_bg, (cx - 400, cy - 100), (cx - 400, cy + 100), normal_dir=(1, 0), color=COLOR_AXIS)
    
    def frame_generator():
        for i in range(num_frames):
            t = t_array[i]
            frame_img = bg.copy()
            draw = ImageDraw.Draw(frame_img)
        
            # Current position
            px = cx + int(re_array[i] * radius)
            py = cy - int(im_array[i] * radius) # -y is up in PIL
        
            # Draw spring from left wall to the mass
            draw_spring(draw, (cx - 400, cy), (px, py), num_coils=8, color=(150, 150, 150))
        
            # Draw vector
            draw.line([(cx, cy), (px, py)], fill=COLOR_VECTOR, width=4)
        
            # Draw mass box at the end
            draw_mass(draw, (px, py), (40, 40), color=COLOR_VECTOR, outline=None)
        
            # Draw projections
            draw.line([(px, py), (px, cy)], fill=COLOR_IM, width=2) # vertical (Im)
            draw.line([(px, py), (cx, py)], fill=COLOR_RE, width=2) # horizontal (Re)
        
            yield np.array(frame_img)
            
    logs = []
    for i, (t, re, im) in enumerate(zip(t_array, re_array, im_array)):
        logs.append([
            {"name": "Angle", "value": f"{t:.2f} rad", "role": "metric", "metric_index": 0},
            {"name": "Re", "value": f"{re:.2f}", "role": "metric", "metric_index": 1},
            {"name": "Im", "value": f"{im:.2f}", "role": "metric", "metric_index": 2}
        ])
        
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
