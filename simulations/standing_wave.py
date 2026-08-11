import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROLE_COLORS = {
    "primary": "#E85D4A",
    "secondary": "#5DA8E8",
    "auxiliary": "#7FAE6B",
    "control": "#D4C24A",
    "static": "#A8B5C2",
}
# Mapped from the desaturated shared palette rather than the old
# COLOR_WAVE1="#FF7F50"/COLOR_WAVE2="#4169E1"/COLOR_SUM="#32FF32" (coral,
# royal blue, neon green).
COLOR_WAVE1 = ROLE_COLORS["primary"]
COLOR_WAVE2 = ROLE_COLORS["secondary"]
COLOR_SUM = ROLE_COLORS["auxiliary"]

def recommended_duration(config: dict) -> float:
    return 15.0

def evaluate_point(t):
    x_antinode = np.pi / 2.0
    x_node = np.pi
    y_anti = np.sin(x_antinode - t) + np.sin(x_antinode + t)
    y_node = np.sin(x_node - t) + np.sin(x_node + t)
    return {"y_antinode": y_anti, "y_node": y_node}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        (0.0, {"y_antinode": 2.0, "y_node": 0.0}, 1e-5),
        (np.pi / 2.0, {"y_antinode": 0.0, "y_node": 0.0}, 1e-5)
    ]
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    width, height = 1080, 800
    center_y = 400
    scale_x = 1080 / (2 * np.pi)
    scale_y = 100.0
    
    x = np.linspace(0, 2 * np.pi, 500)
    
    t_vals = np.linspace(0, duration, num_frames)
    variable_logs = []
    
    # Track the point at x = pi/4 to show distinct y1 and y2 component waves
    x0 = np.pi / 4.0
    aux_y1 = np.sin(x0 - t_vals)
    aux_y2 = np.sin(x0 + t_vals)
    
    print("Generating Standing Wave frames...")
    
    def frame_generator():
        for f in range(num_frames):
            t = t_vals[f]
        
            y1 = np.sin(x - t)
            y2 = np.sin(x + t)
            y_sum = y1 + y2
        
            # Time (t): genuine aggregate, no single corresponding curve ->
            # metric. Amplitude (x=π/2): this literally measures the height
            # of the y_sum curve at that x-position -> a direct 1:1
            # correspondence, so it uses the sum curve's own color
            # ("auxiliary") rather than a generic metric color, per the
            # ROLE_COLORS-vs-METRIC_COLORS correspondence rule (same
            # reasoning as Euler's Re/Im matching their respective curves).
            variable_logs.append([
                {"name": "Time (t)", "value": f"{t:.2f} s", "role": "metric", "metric_index": 0},
                {"name": "Amplitude (x=π/2)", "value": f"{np.sin(x0 - t) + np.sin(x0 + t):.2f}", "role": "auxiliary"},
            ])
        
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)
        
            def draw_wave(y_array, color, lw):
                pts = []
                for i in range(len(x)):
                    px = x[i] * scale_x
                    py = center_y - y_array[i] * scale_y
                    pts.append((px, py))
                draw.line(pts, fill=color, width=lw)
            
            draw_wave(y1, COLOR_WAVE1, 2)
            draw_wave(y2, COLOR_WAVE2, 2)
            draw_wave(y_sum, COLOR_SUM, 5)
        
            yield np.array(img)
        
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
            
    auxiliary_curves = {
        "series": {
            "sin(x-t)": {"data": aux_y1, "color": COLOR_WAVE1},
            "sin(x+t)": {"data": aux_y2, "color": COLOR_WAVE2}
        },
        "time": t_vals
    }
            
    return frame_generator(), variable_logs, auxiliary_curves
