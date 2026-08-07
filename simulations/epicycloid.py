import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOR_ROLE_1 = (255, 50, 50)
COLOR_ROLE_2 = (50, 150, 255)
COLOR_ROLE_3 = (50, 255, 50)
COLOR_TRAIL = (255, 120, 0)

def recommended_duration(config: dict) -> float:
    return 15.0

def evaluate_point(t):
    r = 1.0
    k = 4.0
    x = r * (k + 1) * np.cos(t) - r * np.cos((k + 1) * t)
    y = r * (k + 1) * np.sin(t) - r * np.sin((k + 1) * t)
    return {"x": x, "y": y}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        (0.0, {"x": 4.0, "y": 0.0}, 1e-5),
        (np.pi, {"x": -4.0, "y": 0.0}, 1e-5)
    ]
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    r = 1.0
    k = 4.0
    
    width, height = 1080, 1080
    center_x, center_y = 540, 540
    scale = 100.0
    
    max_t = 2 * np.pi
    t_vals = np.linspace(0, max_t, num_frames)
    
    frames = []
    variable_logs = []
    
    trail = []
    
    print("Generating Epicycloid frames...")
    
    for f in range(num_frames):
        t = t_vals[f]
        
        pt = evaluate_point(t)
        px = center_x + pt["x"] * scale
        py = center_y - pt["y"] * scale
        
        trail.append((px, py))
        
        variable_logs.append({
            "Time (t)": f"{t:.2f} rad",
            "Radius (R)": f"{pt['x']**2 + pt['y']**2:.2f}"
        })
        
        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        
        if len(trail) > 1:
            draw.line(trail, fill=COLOR_ROLE_2 + (255,), width=4)
            
        draw.ellipse([px-6, py-6, px+6, py+6], fill=COLOR_ROLE_1 + (255,))
        
        frames.append(np.array(img))
        
        if (f + 1) % 30 == 0:
            print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frames, variable_logs
