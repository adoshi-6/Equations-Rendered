import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOR_ROLE_1 = (232, 93, 74)     # Primary
COLOR_ROLE_2 = (93, 168, 232)    # Secondary
COLOR_ROLE_3 = (127, 174, 107)   # Auxiliary
COLOR_TRAIL = (93, 168, 232)     # Secondary for trails

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

    max_t = 2 * np.pi
    t_vals = np.linspace(0, max_t, num_frames)
    # PRECOMPUTE BOUNDS (Section 3.6)
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for f in range(num_frames):
        t = t_vals[f]
        rc_cx = (k + 1) * r * np.cos(t)
        rc_cy = (k + 1) * r * np.sin(t)
        min_x = min(min_x, rc_cx - r)
        max_x = max(max_x, rc_cx + r)
        min_y = min(min_y, rc_cy - r)
        max_y = max(max_y, rc_cy + r)
        
    range_x = max(max_x - min_x, 1e-3)
    range_y = max(max_y - min_y, 1e-3)
    margin_x = range_x * 0.10
    margin_y = range_y * 0.10
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    
    scale_x = 1080 / (max_x - min_x)
    scale_y = 1080 / (max_y - min_y)
    scale = float(min(scale_x, scale_y))
    center_x = 1080 / 2 - ((min_x + max_x) / 2) * scale
    center_y = 1080 / 2 + ((min_y + max_y) / 2) * scale

    

    variable_logs = []
    
    trail = []
    
    print("Generating Epicycloid frames...")
    
    def frame_generator():
        for f in range(num_frames):
            t = t_vals[f]
        
            pt = evaluate_point(t)
            px = center_x + pt["x"] * scale
            py = center_y - pt["y"] * scale
        
            trail.append((px, py))
        
            # NOTE: previously this field was labeled "Radius (R)" but computed
            # pt['x']**2 + pt['y']**2 — the SQUARED distance from origin to the
            # traced point (missing sqrt, and varying every frame). That is
            # neither the epicycloid's actual fixed parameter R (=k*r, a
            # constant, drawn as the large annotation circle) nor a true
            # distance. Fixed to report the actual fixed R parameter, tagged
            # "auxiliary" since it corresponds to the green annotation circle,
            # not the red traced-point marker ("primary").
            variable_logs.append([
                {"name": "Time (t)", "value": f"{t:.2f} rad", "role": "metric", "metric_index": 0},
                {"name": "Fixed Radius (R)", "value": f"{k * r:.2f}", "role": "auxiliary"}
            ])
        
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)
            
            R_radius = k * r * scale
            r_radius = r * scale
            
            # Pedagogical Annotation: Fixed Circle (R)
            draw.ellipse([center_x - R_radius, center_y - R_radius, center_x + R_radius, center_y + R_radius], outline=COLOR_ROLE_3 + (150,), width=2)
            
            # Pedagogical Annotation: Rolling Circle (r)
            rc_x = center_x + (k + 1) * r * scale * np.cos(t)
            rc_y = center_y - (k + 1) * r * scale * np.sin(t)
            draw.ellipse([rc_x - r_radius, rc_y - r_radius, rc_x + r_radius, rc_y + r_radius], outline=COLOR_ROLE_3 + (255,), width=2)
            
            # Pedagogical Annotation: Radii lines and labels
            try:
                from renderer import load_italic_font
                ann_font = load_italic_font(28)
            except ImportError:
                ann_font = None
                
            # R line from origin to fixed circle edge
            R_end_x = center_x + R_radius * np.cos(t)
            R_end_y = center_y - R_radius * np.sin(t)
            draw.line([(center_x, center_y), (R_end_x, R_end_y)], fill=COLOR_ROLE_3 + (150,), width=2)
            # Label R at midpoint
            if ann_font:
                draw.text(((center_x + R_end_x) // 2 + 5, (center_y + R_end_y) // 2 - 20), "R", fill=COLOR_ROLE_3 + (255,), font=ann_font)
            else:
                draw.text(((center_x + R_end_x) // 2 + 5, (center_y + R_end_y) // 2 - 20), "R", fill=COLOR_ROLE_3 + (255,))
            
            # r line from rc_center to px, py
            draw.line([(rc_x, rc_y), (px, py)], fill=COLOR_ROLE_3 + (255,), width=2)
            # Label r at midpoint
            if ann_font:
                draw.text(((rc_x + px) // 2 + 5, (rc_y + py) // 2 - 20), "r", fill=COLOR_ROLE_3 + (255,), font=ann_font)
            else:
                draw.text(((rc_x + px) // 2 + 5, (rc_y + py) // 2 - 20), "r", fill=COLOR_ROLE_3 + (255,))
            
            if len(trail) > 1:
                draw.line(trail, fill=COLOR_TRAIL + (255,), width=4)
            
            draw.ellipse([px-6, py-6, px+6, py+6], fill=COLOR_ROLE_1 + (255,))
        
            # We want RGB for compositing in renderer
            yield np.array(img.convert("RGB"))
        
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frame_generator(), variable_logs, None
