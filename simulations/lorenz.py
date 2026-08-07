import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend and integrators can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp
from integrators import rk4_step

def lorenz_derivs(state, t, sigma, rho, beta):
    """
    Lorenz system equations of motion.
    state shape: (num_trajectories, 3) where columns are [x, y, z]
    """
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    deriv = xp.zeros_like(state)
    deriv[:, 0] = dx
    deriv[:, 1] = dy
    deriv[:, 2] = dz
    
    return deriv

# Universal Palette
COLOR_ROLE_1 = (255, 50, 50)     # Bright Red (Leading particles)
COLOR_ROLE_2 = (50, 150, 255)    # Electric Blue
COLOR_ROLE_3 = (50, 255, 50)     # Neon Green
COLOR_TRAIL = (255, 120, 0)      # Orange (Paths/Trails)

from duration_utils import BoundingBoxPlateauDetector

def recommended_duration(config: dict) -> float:
    """
    Computes duration by running a coarse headless simulation to see when 
    the Lorenz system diverges (chaos bloom) OR reaches a spatial plateau.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    num_trajectories = 100
    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 10.0 + xp.linspace(-1e-4, 1e-4, num_trajectories)
    state[:, 1] = 10.0
    state[:, 2] = 10.0
    
    dt = 0.05
    max_t = 30.0
    t = 0.0
    
    detector = BoundingBoxPlateauDetector(patience_steps=50, dt=dt, rel_tolerance=1e-3)
    
    while t < max_t:
        state = rk4_step(lorenz_derivs, state, t, dt, sigma, rho, beta)
        t += dt
        
        coords = state[:, 0:3]
        plateaued = detector.check(coords)
        
        # Check divergence of x coordinate
        std_dev = float(xp.std(state[:, 0]))
        if plateaued or std_dev > 5.0:
            reason = "spatial extent plateaued" if plateaued else "divergence threshold (5.0) crossed"
            print(f"Lorenz {reason} at {t:.1f}s. Adding 3s buffer.")
            return t + 3.0
            
    print(f"Lorenz stopping criteria not met within {max_t}s.")
    return max_t

def simulate_headless(config: dict):
    duration = config.get("duration", 2.0)
    dt_divider = config.get("dt_divider", 1)
    
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    num_trajectories = 100
    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 10.0 + xp.linspace(-1e-4, 1e-4, num_trajectories)
    state[:, 1] = 10.0
    state[:, 2] = 10.0
    
    dt = 0.05 / dt_divider
    max_t = duration
    t = 0.0
    
    states = []
    times = []
    
    while t <= max_t + 1e-5:
        states.append(state.get() if hasattr(state, "get") else np.asarray(state))
        times.append(t)
        state = rk4_step(lorenz_derivs, state, t, dt, sigma, rho, beta)
        t += dt
        
    return np.array(times), states

def get_state_variables(states):
    # states is a list of arrays of shape (num_trajectories, 3)
    # We want to concatenate all times to find global min/max
    stacked = np.stack(states, axis=0) # shape (T, num_trajectories, 3)
    return {
        "x": stacked[:, :, 0],
        "y": stacked[:, :, 1],
        "z": stacked[:, :, 2]
    }

TEST_SPEC = {
    "category": "bounded_region",
    "expected_bounds": {
        "x": {"min": -20.0, "max": 20.0},
        "y": {"min": -25.0, "max": 25.0},
        "z": {"min": 0.0, "max": 50.0}
    }
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Lorenz Attractor simulation.
    """
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    # Lorenz system constants
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    num_trajectories = 200
    
    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 10.0 + xp.linspace(-1e-4, 1e-4, num_trajectories)
    state[:, 1] = 10.0
    state[:, 2] = 10.0
    
    dt_frame = 1.0 / fps
    n_substeps = 15
    dt = dt_frame / n_substeps
    
    width, height = 1080, 1080
    center_x, center_y = 540, 580
    scale = 18.0
    
    trail_history = []
    max_trail_len = 100
    
    frames = []
    variable_logs = []
    
    def get_trail_color(i, n):
        t = i / max(1, n - 1)
        r = int(255 * (1 - t))
        g = int(255 * t)
        b = 255
        return (r, g, b)
        
    t_curr = 0.0
    for f in range(num_frames):
        for _ in range(n_substeps):
            state = rk4_step(lorenz_derivs, state, t_curr, dt, sigma, rho, beta)
            t_curr += dt
            
        st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
        x, y, z = st_cpu[:, 0], st_cpu[:, 1], st_cpu[:, 2]
        
        # Record variable log
        std_dev = float(np.std(x))
        variable_logs.append({
            "Divergence (σ)": f"{std_dev:.3f}",
            "Avg X": f"{float(np.mean(x)):.2f}"
        })
        
        px = center_x + x * scale
        py = center_y - (z - 25) * scale
        
        trail_history.append((px.copy(), py.copy()))
        if len(trail_history) > max_trail_len:
            trail_history.pop(0)
            
        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        
        history_len = len(trail_history)
        if history_len > 1:
            for h in range(1, history_len):
                opacity_factor = h / history_len
                alpha = int(140 * opacity_factor**1.8)
                
                prev_x, prev_y = trail_history[h-1]
                curr_x, curr_y = trail_history[h]
                
                for i in range(num_trajectories):
                    draw.line(
                        [(prev_x[i], prev_y[i]), (curr_x[i], curr_y[i])],
                        fill=get_trail_color(i, num_trajectories) + (alpha,),
                        width=2
                    )
                    
        for i in range(num_trajectories):
            draw.ellipse([px[i]-3, py[i]-3, px[i]+3, py[i]+3], fill=COLOR_ROLE_1 + (255,))
            
        frames.append(np.array(img.convert("RGB")))
        
    return frames, variable_logs
