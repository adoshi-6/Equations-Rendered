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

# Desaturated shared palette (see CONVENTIONS.md Section 3.3)
ROLE_COLORS = {
    "primary": (232, 93, 74),      # #E85D4A
    "secondary": (93, 168, 232),   # #5DA8E8
    "auxiliary": (127, 174, 107),  # #7FAE6B
    "control": (212, 194, 74),     # #D4C24A
    "static": (168, 181, 194),     # #A8B5C2
}
METRIC_COLORS = [
    (232, 144, 93),   # #E8905D amber
    (184, 127, 201),  # #B87FC9 dusty purple
    (201, 127, 160),  # #C97FA0 muted rose
    (127, 201, 176),  # #7FC9B0 muted teal
]

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
    },
    "also_run": ["trend_assertions"],
    "trend_assertions": {
        "Divergence (σ)": "monotonic_increase"
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
    
    dt_frame = 1.0 / fps
    n_substeps = 15
    dt = dt_frame / n_substeps

    width, height = 1080, 1080

    # PRECOMPUTE BOUNDS (Section 3.6): run the actual full-duration
    # trajectory once, headless, to find the true x/z extent over the
    # ENTIRE render — not a hardcoded center/scale that happened to look
    # right for one particular duration/parameter set. Every hardcoded
    # scale=18.0/center=(540,580) in the pre-migration version of this file
    # was tuned by eye once and never re-verified.
    bounds_state = xp.zeros((num_trajectories, 3))
    bounds_state[:, 0] = 10.0 + xp.linspace(-1e-4, 1e-4, num_trajectories)
    bounds_state[:, 1] = 10.0
    bounds_state[:, 2] = 10.0
    t_b = 0.0
    x_min = x_max = z_min = z_max = None
    for f in range(num_frames):
        for _ in range(n_substeps):
            bounds_state = rk4_step(lorenz_derivs, bounds_state, t_b, dt, sigma, rho, beta)
            t_b += dt
        st_cpu = bounds_state.get() if hasattr(bounds_state, "get") else np.asarray(bounds_state)
        bx, bz = st_cpu[:, 0], st_cpu[:, 2]
        fx_min, fx_max = float(np.min(bx)), float(np.max(bx))
        fz_min, fz_max = float(np.min(bz)), float(np.max(bz))
        x_min = fx_min if x_min is None else min(x_min, fx_min)
        x_max = fx_max if x_max is None else max(x_max, fx_max)
        z_min = fz_min if z_min is None else min(z_min, fz_min)
        z_max = fz_max if z_max is None else max(z_max, fz_max)

    # Fit the true (x, z) extent into the canvas with a 10% margin on all sides
    x_range = max(x_max - x_min, 1e-6)
    z_range = max(z_max - z_min, 1e-6)
    margin = 0.10
    scale = min(
        width * (1 - 2 * margin) / x_range,
        height * (1 - 2 * margin) / z_range,
    )
    center_x = width / 2 - (x_min + x_max) / 2 * scale
    # NOTE: the per-frame pixel transform below is `py = center_y - (z - 25) * scale`
    # (the "-25" is a pre-existing offset baked into that formula). Must be
    # accounted for here too, or the vertical centering is off by 25*scale
    # pixels — confirmed as a real bug during verification: without this
    # correction, the attractor rendered cramped into the bottom of the
    # frame, cut off at the top.
    z_mid = (z_min + z_max) / 2
    center_y = height / 2 + (z_mid - 25) * scale

    # Real simulation state (separate from the bounds pre-pass above)
    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 10.0 + xp.linspace(-1e-4, 1e-4, num_trajectories)
    state[:, 1] = 10.0
    state[:, 2] = 10.0

    trail_history = []
    # Trail grows across the whole render rather than a small fixed window —
    # a short fixed-length trail (previously 100 frames, ~3.3s) would only
    # ever show a partial loop, not the classic full two-lobe Lorenz
    # butterfly shape. Matches the fix already verified for Rössler, which
    # had the same problem (identical mid/end frames due to a hardcoded cap
    # regardless of how far the render had actually progressed).
    max_trail_len = num_frames

    variable_logs = []
    
    def get_trail_color(i, n):
        # Desaturated gradient between `secondary` and `auxiliary` across the
        # trajectory ensemble, replacing the old pure red-green-blue gradient.
        t = i / max(1, n - 1)
        c1, c2 = ROLE_COLORS["secondary"], ROLE_COLORS["auxiliary"]
        return tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3))
        
    def frame_generator():
        t_curr = 0.0
        for f in range(num_frames):
            nonlocal state
            for _ in range(n_substeps):
                state = rk4_step(lorenz_derivs, state, t_curr, dt, sigma, rho, beta)
                t_curr += dt
                
            st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
            x, y, z = st_cpu[:, 0], st_cpu[:, 1], st_cpu[:, 2]
            
            # Record variable log. Both values are genuine aggregates across
            # all 200 trajectories (no single corresponding colored curve),
            # so METRIC_COLORS applies per the ROLE_COLORS-vs-METRIC_COLORS
            # correspondence rule (CONVENTIONS.md Section 3.3).
            std_dev = float(np.std(x))
            variable_logs.append([
                {"name": "Divergence (σ)", "value": f"{std_dev:.3f}", "role": "metric", "metric_index": 0},
                {"name": "Avg X", "value": f"{float(np.mean(x)):.2f}", "role": "metric", "metric_index": 1},
            ])
            
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
                draw.ellipse([px[i]-3, py[i]-3, px[i]+3, py[i]+3], fill=ROLE_COLORS["primary"] + (255,))
                
            yield np.array(img.convert("RGB"))
            
    return frame_generator(), variable_logs, None
