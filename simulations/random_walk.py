import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

COLOR_ROLE_1 = (255, 50, 50)
COLOR_ROLE_2 = (50, 150, 255)
COLOR_ROLE_3 = (50, 255, 50)
COLOR_TRAIL = (255, 120, 0)

def recommended_duration(config: dict) -> float:
    return 15.0

def simulate_headless(config: dict):
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    
    num_particles = 10000
    D = 1.0
    
    dt = 1.0 / fps
    max_t = duration
    t = 0.0
    
    state = xp.zeros((num_particles, 2))
    
    states = []
    times = []
    
    while t <= max_t + 1e-5:
        st_val = state.get() if hasattr(state, "get") else np.asarray(state)
        states.append(st_val.copy())
        times.append(t)
        
        # Random walk step
        step_std = xp.sqrt(2 * D * dt)
        step = xp.random.normal(0, step_std, size=(num_particles, 2))
        state += step
        
        t += dt
        
    return np.array(times), states

def get_ensemble_stats(config):
    t_array, states = simulate_headless(config)
    
    var_x = [np.var(st[:, 0]) for st in states]
    var_y = [np.var(st[:, 1]) for st in states]
    
    # Fit linear slope (y = mx + c)
    slope_x, _ = np.polyfit(t_array, var_x, 1)
    slope_y, _ = np.polyfit(t_array, var_y, 1)
    
    return {
        "slope_x": float(slope_x),
        "slope_y": float(slope_y)
    }

TEST_SPEC = {
    "category": "ensemble_stats",
    "expected_stats": {
        "slope_x": 2.0,
        "slope_y": 2.0
    },
    "tolerance_percent": 15.0
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    num_particles = 3000
    D = 1.0
    
    dt = 1.0 / fps
    
    width, height = 1080, 1080
    center_x, center_y = 540, 540
    scale = 35.0
    
    state = xp.zeros((num_particles, 2))
    
    frames = []
    variable_logs = []
    
    print("Generating Random Walk frames...")
    
    t_curr = 0.0
    
    for f in range(num_frames):
        step_std = xp.sqrt(2 * D * dt)
        step = xp.random.normal(0, step_std, size=(num_particles, 2))
        state += step
        t_curr += dt
        
        st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
        
        var_x = float(np.var(st_cpu[:, 0]))
        var_y = float(np.var(st_cpu[:, 1]))
        
        variable_logs.append({
            "Time (t)": f"{t_curr:.2f} s",
            "Variance (X)": f"{var_x:.2f}",
            "Variance (Y)": f"{var_y:.2f}"
        })
        
        px = center_x + st_cpu[:, 0] * scale
        py = center_y - st_cpu[:, 1] * scale
        
        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        
        for i in range(num_particles):
            draw.ellipse([px[i]-2, py[i]-2, px[i]+2, py[i]+2], fill=COLOR_ROLE_2 + (150,))
            
        frames.append(np.array(img))
        
        if (f + 1) % 30 == 0:
            print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frames, variable_logs
