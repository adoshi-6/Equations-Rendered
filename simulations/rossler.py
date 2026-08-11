import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp
from integrators import rk4_step
from duration_utils import BoundingBoxPlateauDetector

def rossler_derivs(state, t, a, b, c):
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]
    
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    
    deriv = xp.zeros_like(state)
    deriv[:, 0] = dx
    deriv[:, 1] = dy
    deriv[:, 2] = dz
    return deriv

ROLE_COLORS = {
    "primary": (232, 93, 74),
    "secondary": (93, 168, 232),
    "auxiliary": (127, 174, 107),
    "control": (212, 194, 74),
    "static": (168, 181, 194),
}
METRIC_COLORS = [
    (232, 144, 93),
    (184, 127, 201),
    (201, 127, 160),
    (127, 201, 176),
]

def recommended_duration(config: dict) -> float:
    a, b, c = 0.2, 0.2, 5.7
    num_trajectories = 100
    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 1.0 + xp.linspace(-1e-3, 1e-3, num_trajectories)
    state[:, 1] = 1.0
    state[:, 2] = 1.0
    
    dt = 0.05
    max_t = 26.666
    t = 0.0
    
    detector = BoundingBoxPlateauDetector(patience_steps=100, dt=dt, rel_tolerance=1e-3)
    
    while t < max_t:
        state = rk4_step(rossler_derivs, state, t, dt, a, b, c)
        t += dt
        
        coords = state[:, 0:3]
        plateaued = detector.check(coords)
        
        std_dev = float(xp.std(state[:, 0]))
        if plateaued or std_dev > 5.0:
            return 26.666
            
    return 26.666

def simulate_headless(config: dict):
    duration = config.get("duration", 2.0)
    dt_divider = config.get("dt_divider", 1)
    
    a, b, c = 0.2, 0.2, 5.7
    num_trajectories = 100
    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 1.0 + xp.linspace(-1e-3, 1e-3, num_trajectories)
    state[:, 1] = 1.0
    state[:, 2] = 1.0
    
    dt = 0.05 / dt_divider
    max_t = duration
    t = 0.0
    
    states = []
    times = []
    
    while t <= max_t + 1e-5:
        states.append(state.get() if hasattr(state, "get") else np.asarray(state))
        times.append(t)
        state = rk4_step(rossler_derivs, state, t, dt, a, b, c)
        t += dt
        
    return np.array(times), states

def get_state_variables(states):
    stacked = np.stack(states, axis=0)
    return {
        "x": stacked[:, :, 0],
        "y": stacked[:, :, 1],
        "z": stacked[:, :, 2]
    }

TEST_SPEC = {
    "category": "bounded_region",
    "expected_bounds": {
        "x": {"min": -30.0, "max": 30.0},
        "y": {"min": -30.0, "max": 30.0},
        "z": {"min": 0.0, "max": 100.0}
    },
    "also_run": ["convergence_dt", "trend_assertions"],
    # Divergence (std dev of x across the trajectory ensemble) should trend
    # upward over the render as initially-clustered trajectories separate —
    # matches the empirically-verified behavior from manual review (divergence
    # 0.00085 -> 0.00205 -> 0.00253 across early/mid/late frames of a real
    # render). Max Z is intentionally NOT asserted monotonic here since it's a
    # genuinely chaotic, non-monotonic quantity — asserting that would produce
    # false failures on correct renders.
    "trend_assertions": {
        "Divergence (σ)": "monotonic_increase"
    }
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    a, b, c = 0.2, 0.2, 5.7
    num_trajectories = 200
    
    dt_frame = 1.0 / fps
    n_substeps = 15
    dt = dt_frame / n_substeps

    width, height = 1080, 1080

    # PRECOMPUTE BOUNDS (Section 3.6): real full-duration headless pass,
    # replacing the previously-hardcoded center=(540,580)/scale=18.0 (values
    # shared byte-for-byte with lorenz.py, suggesting copy-paste rather than
    # individual tuning, and never re-verified against actual render output).
    # Projection here uses (x, y) only — z is tracked separately for "Max Z"
    # but isn't part of the 2D projection, so only x/y extent is needed.
    bounds_state = xp.zeros((num_trajectories, 3))
    bounds_state[:, 0] = 1.0 + xp.linspace(-1e-3, 1e-3, num_trajectories)
    bounds_state[:, 1] = 1.0
    bounds_state[:, 2] = 1.0
    t_b = 0.0
    x_min = x_max = y_min = y_max = None
    for f in range(num_frames):
        for _ in range(n_substeps):
            bounds_state = rk4_step(rossler_derivs, bounds_state, t_b, dt, a, b, c)
            t_b += dt
        st_cpu = bounds_state.get() if hasattr(bounds_state, "get") else np.asarray(bounds_state)
        bx, by = st_cpu[:, 0], st_cpu[:, 1]
        fx_min, fx_max = float(np.min(bx)), float(np.max(bx))
        fy_min, fy_max = float(np.min(by)), float(np.max(by))
        x_min = fx_min if x_min is None else min(x_min, fx_min)
        x_max = fx_max if x_max is None else max(x_max, fx_max)
        y_min = fy_min if y_min is None else min(y_min, fy_min)
        y_max = fy_max if y_max is None else max(y_max, fy_max)

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    margin = 0.10
    scale = min(
        width * (1 - 2 * margin) / x_range,
        height * (1 - 2 * margin) / y_range,
    )
    center_x = width / 2 - (x_min + x_max) / 2 * scale
    center_y = height / 2 + (y_min + y_max) / 2 * scale

    state = xp.zeros((num_trajectories, 3))
    state[:, 0] = 1.0 + xp.linspace(-1e-3, 1e-3, num_trajectories)
    state[:, 1] = 1.0
    state[:, 2] = 1.0
    
    trail_history = []
    # Generalized to scale with actual render length (was previously a
    # verified-correct but hardcoded 800, tuned for one specific duration).
    max_trail_len = num_frames
    
    variable_logs = []
    
    def get_trail_color(i, n):
        t = i / max(1, n - 1)
        c1, c2 = ROLE_COLORS["secondary"], ROLE_COLORS["primary"]
        return tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3))
    
    def frame_generator():
        print("Simulating Rössler Attractor trajectories...")
        t_curr = 0.0
        
        for f in range(num_frames):
            nonlocal state
            for _ in range(n_substeps):
                state = rk4_step(rossler_derivs, state, t_curr, dt, a, b, c)
                t_curr += dt
                
            st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
            std_dev = float(np.std(st_cpu[:, 0]))
            max_z = float(np.max(st_cpu[:, 2]))
            
            # Both are genuine aggregates across all 200 trajectories — no
            # single corresponding colored curve — so METRIC_COLORS applies.
            variable_logs.append([
                {"name": "Max Z", "value": f"{max_z:.2f}", "role": "metric", "metric_index": 0},
                {"name": "Divergence (σ)", "value": f"{std_dev:.5f}", "role": "metric", "metric_index": 1},
            ])
            
            # Projection: use x and y
            px = center_x + st_cpu[:, 0] * scale
            py = center_y - st_cpu[:, 1] * scale
            
            trail_history.append((px.copy(), py.copy()))
            if len(trail_history) > max_trail_len:
                trail_history.pop(0)
                
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)
            
            history_len = len(trail_history)
            if history_len > 1:
                for h in range(1, history_len):
                    opacity_factor = h / history_len
                    alpha = int(120 * opacity_factor**1.5)
                    
                    prev_x, prev_y = trail_history[h-1]
                    curr_x, curr_y = trail_history[h]
                    
                    for i in range(num_trajectories):
                        draw.line(
                            [(prev_x[i], prev_y[i]), (curr_x[i], curr_y[i])],
                            fill=get_trail_color(i, num_trajectories) + (alpha,),
                            width=2
                        )
                        
            for i in range(num_trajectories):
                draw.ellipse(
                    [px[i]-3, py[i]-3, px[i]+3, py[i]+3],
                    fill=ROLE_COLORS["auxiliary"] + (255,)
                )
                
            yield np.array(img.convert("RGB"))
            
            if (f + 1) % 30 == 0:
                print(f"Processed frame {f + 1}/{num_frames}...")
                
    return frame_generator(), variable_logs, None
