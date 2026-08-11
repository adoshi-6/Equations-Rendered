import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

ROLE_COLORS = {
    "primary": (232, 93, 74),
    "secondary": (93, 168, 232),
    "auxiliary": (127, 174, 107),
    "control": (212, 194, 74),
    "static": (168, 181, 194),
}

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

    # Bounds (Section 3.6): unlike the chaotic-ODE simulations, this
    # diffusion process has an EXACT analytic growth law — variance grows
    # linearly as 2*D*t (this is literally what TEST_SPEC's ensemble_stats
    # check verifies against real simulation output). That means the correct
    # scale can be derived directly, without an expensive numerical pre-pass:
    # std at the end of the render is sqrt(2*D*duration), and with 3000
    # particles sampled from a Gaussian, outliers can reasonably reach ~4.5
    # standard deviations. Previously this was a flat hardcoded scale=35.0
    # that happened to fit at the one duration it was tuned for (15s) purely
    # by coincidence — a longer render would have pushed outlier particles
    # toward/past the frame edge.
    sigma_max = float(np.sqrt(2 * D * duration))
    half_extent = 4.5 * sigma_max
    margin = 0.10
    scale = (min(width, height) * (1 - 2 * margin)) / (2 * half_extent)
    center_x, center_y = width / 2, height / 2

    state = xp.zeros((num_particles, 2))
    variable_logs = []
    
    print("Generating Random Walk frames...")
    
    t_curr = 0.0
    
    def frame_generator():
        nonlocal state, t_curr
        for f in range(num_frames):
            step_std = xp.sqrt(2 * D * dt)
            step = xp.random.normal(0, step_std, size=(num_particles, 2))
            state += step
            t_curr += dt
        
            st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
        
            var_x = float(np.var(st_cpu[:, 0]))
            var_y = float(np.var(st_cpu[:, 1]))
        
            # All three are genuine aggregates across the 3000-particle
            # ensemble (no single corresponding colored curve) -> metric,
            # each with a distinct METRIC_COLORS index.
            variable_logs.append([
                {"name": "Time (t)", "value": f"{t_curr:.2f} s", "role": "metric", "metric_index": 0},
                {"name": "Variance (X)", "value": f"{var_x:.2f}", "role": "metric", "metric_index": 1},
                {"name": "Variance (Y)", "value": f"{var_y:.2f}", "role": "metric", "metric_index": 2},
            ])
        
            px = center_x + st_cpu[:, 0] * scale
            py = center_y - st_cpu[:, 1] * scale
        
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img, "RGBA")
        
            for i in range(num_particles):
                draw.ellipse([px[i]-2, py[i]-2, px[i]+2, py[i]+2], fill=ROLE_COLORS["secondary"] + (150,))
            
            yield np.array(img)
        
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
            
    return frame_generator(), variable_logs, None
