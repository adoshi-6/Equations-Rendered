import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

# Universal Palette
COLOR_ROLE_1 = (255, 50, 50)     # Bright Red
COLOR_ROLE_2 = (50, 150, 255)    # Electric Blue
COLOR_ROLE_3 = (50, 255, 50)     # Neon Green
COLOR_TRAIL = (255, 120, 0)      # Orange

def evaluate_point(inp):
    theta = inp["theta"]
    R, r, d = 300.0, 180.0, 150.0
    ratio = (R - r) / r
    x = (R - r) * np.cos(theta) + d * np.cos(ratio * theta)
    y = (R - r) * np.sin(theta) - d * np.sin(ratio * theta)
    return {"x": float(x), "y": float(y)}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        ({"theta": 0.0}, {"x": 270.0, "y": 0.0}, 1e-6),
        ({"theta": np.pi}, {"x": -120.0 + 150.0 * np.cos(2/3 * np.pi), "y": 0.0 - 150.0 * np.sin(2/3 * np.pi)}, 1e-4),
    ],
}

def recommended_duration(config: dict) -> float:
    """
    Computes duration for drawing the complete spirograph pattern.
    10 seconds provides a smooth drawing experience.
    """
    return 10.0

def get_gradient_color(t):
    """
    Generates a gradient using the universal palette (Red -> Blue -> Green).
    t is the normalized position along the curve [0, 1].
    """
    if t < 0.5:
        f = t * 2.0
        r = int(COLOR_ROLE_1[0] * (1.0 - f) + COLOR_ROLE_2[0] * f)
        g = int(COLOR_ROLE_1[1] * (1.0 - f) + COLOR_ROLE_2[1] * f)
        b = int(COLOR_ROLE_1[2] * (1.0 - f) + COLOR_ROLE_2[2] * f)
    else:
        f = (t - 0.5) * 2.0
        r = int(COLOR_ROLE_2[0] * (1.0 - f) + COLOR_ROLE_3[0] * f)
        g = int(COLOR_ROLE_2[1] * (1.0 - f) + COLOR_ROLE_3[1] * f)
        b = int(COLOR_ROLE_2[2] * (1.0 - f) + COLOR_ROLE_3[2] * f)
    return (r, g, b)

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Spirograph drawing animation.
    """
    duration = config.get("duration", 10.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 1080
    center_x, center_y = 540, 540
    
    R = 300.0
    r = 180.0
    d = 150.0
    
    max_theta = 12.0 * xp.pi
    
    num_points_full = 1000
    theta_full = xp.linspace(0, max_theta, num_points_full)
    
    ratio = (R - r) / r
    x_full = (R - r) * xp.cos(theta_full) + d * xp.cos(ratio * theta_full)
    y_full = (R - r) * xp.sin(theta_full) - d * xp.sin(ratio * theta_full)
    
    px_full = center_x + x_full
    py_full = center_y - y_full
    
    if hasattr(px_full, "get"):
        px_full_cpu = px_full.get()
        py_full_cpu = py_full.get()
    else:
        px_full_cpu = np.asarray(px_full)
        py_full_cpu = np.asarray(py_full)
    variable_logs = []
    
    print("Generating Spirograph frames...")
    
    def frame_generator():
        for f in range(num_frames):
            progress = f / max(1, num_frames - 1)
            current_idx = int(progress * (num_points_full - 1))
        
            theta_t = progress * max_theta
        
            # Log variable
            variable_logs.append({
                "Rotor θ": f"{float(theta_t):.2f} rad",
                "Ratio R/r": f"{R/r:.2f}"
            })
        
            cx = center_x + (R - r) * np.cos(theta_t)
            cy = center_y - (R - r) * np.sin(theta_t)
        
            pen_x = cx + d * np.cos(ratio * theta_t)
            pen_y = cy - d * np.sin(ratio * theta_t)
        
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)
        
            draw.ellipse([center_x - R, center_y - R, center_x + R, center_y + R], outline=(100, 100, 100, 255), width=2)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(150, 150, 150, 255), width=1)
        
            draw.line([(cx, cy), (pen_x, pen_y)], fill=(200, 200, 200, 255), width=2)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(200, 200, 200, 255))
        
            if current_idx > 1:
                for k in range(1, current_idx + 1):
                    p_norm = k / (num_points_full - 1)
                    color = get_gradient_color(p_norm)
                
                    draw.line(
                        [(px_full_cpu[k-1], py_full_cpu[k-1]), (px_full_cpu[k], py_full_cpu[k])],
                        fill=color + (255,),
                        width=3
                    )
                
            draw.ellipse([pen_x - 5, pen_y - 5, pen_x + 5, pen_y + 5], fill=(255, 255, 255, 255), outline=COLOR_ROLE_1 + (255,), width=2)
        
            yield np.array(img.convert("RGB"))
        
    return frame_generator(), variable_logs, None
