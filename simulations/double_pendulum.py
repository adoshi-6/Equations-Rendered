import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend and integrators can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp
from integrators import rk4_step

def double_pendulum_derivs(state, t, L1, L2, m1, m2, g):
    """
    State vector: [theta1, theta2, w1, w2]
    state shape: (num_trajectories, 4)
    """
    theta1 = state[:, 0]
    theta2 = state[:, 1]
    w1 = state[:, 2]
    w2 = state[:, 3]
    
    delta = theta1 - theta2
    
    mu = 1.0 + m1 / m2
    cos_d = xp.cos(delta)
    sin_d = xp.sin(delta)
    
    # Equation for theta1 acceleration (alpha1)
    den1 = L1 * (mu - cos_d**2)
    num1 = g * (xp.sin(theta2) * cos_d - mu * xp.sin(theta1)) - (L2 * w2**2 + L1 * w1**2 * cos_d) * sin_d
    alpha1 = num1 / den1
    
    # Equation for theta2 acceleration (alpha2)
    den2 = L2 * (mu - cos_d**2)
    num2 = g * mu * (xp.sin(theta1) * cos_d - xp.sin(theta2)) + (mu * L1 * w1**2 + L2 * w2**2 * cos_d) * sin_d
    alpha2 = num2 / den2
    
    # Derivatives
    deriv = xp.zeros_like(state)
    deriv[:, 0] = w1
    deriv[:, 1] = w2
    deriv[:, 2] = alpha1
    deriv[:, 3] = alpha2
    
    return deriv

# Universal Palette
COLOR_ROLE_1 = (232, 93, 74)     # Primary (Mass 1 / Rod 1)
COLOR_ROLE_2 = (93, 168, 232)    # Secondary (Mass 2 / Rod 2)
COLOR_ROLE_3 = (127, 174, 107)   # Auxiliary (Pivot)
COLOR_TRAIL = (93, 168, 232)     # Match Secondary for Trails

from duration_utils import BoundingBoxPlateauDetector

def recommended_duration(config: dict) -> float:
    """
    Computes duration by running a coarse headless simulation to see when 
    the system diverges (chaos bloom) OR reaches a spatial plateau.
    """
    L1, L2, m1, m2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    num_trajectories = 100
    state = xp.zeros((num_trajectories, 4))
    state[:, 0] = 2.0 + xp.linspace(-1e-5, 1e-5, num_trajectories)
    state[:, 1] = 2.0
    
    dt = 0.05
    max_t = 30.0
    t = 0.0
    
    detector = BoundingBoxPlateauDetector(patience_steps=50, dt=dt, rel_tolerance=1e-3)
    
    while t < max_t:
        state = rk4_step(double_pendulum_derivs, state, t, dt, L1, L2, m1, m2, g)
        t += dt
        
        # Calculate x2, y2 coordinates for plateau detection
        theta1 = state[:, 0]
        theta2 = state[:, 1]
        x1 = L1 * xp.sin(theta1)
        y1 = -L1 * xp.cos(theta1)
        x2 = x1 + L2 * xp.sin(theta2)
        y2 = y1 - L2 * xp.cos(theta2)
        
        coords = xp.stack([x2, y2], axis=-1)
        plateaued = detector.check(coords)
        
        # Check divergence of theta2
        std_dev = float(xp.std(theta2))
        if plateaued or std_dev > 1.0:
            reason = "spatial extent plateaued" if plateaued else "divergence threshold (1.0 rad) crossed"
            print(f"Double Pendulum {reason} at {t:.1f}s. Adding 2s buffer.")
            return t + 2.0
            
    print(f"Double Pendulum stopping criteria not met within {max_t}s.")
    return max_t


def simulate_headless(config: dict):
    duration = config.get("duration", 2.0)
    dt_divider = config.get("dt_divider", 1)
    
    L1, L2, m1, m2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    num_trajectories = 100
    state = xp.zeros((num_trajectories, 4))
    state[:, 0] = 2.0 + xp.linspace(-1e-5, 1e-5, num_trajectories)
    state[:, 1] = 2.0
    
    dt = 0.05 / dt_divider
    max_t = duration
    t = 0.0
    
    states = []
    times = []
    
    while t <= max_t + 1e-5:
        states.append(state.get() if hasattr(state, "get") else np.asarray(state))
        times.append(t)
        state = rk4_step(double_pendulum_derivs, state, t, dt, L1, L2, m1, m2, g)
        t += dt
        
    return np.array(times), states

def compute_energy(state_cpu):
    L1, L2, m1, m2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    theta1 = state_cpu[:, 0]
    theta2 = state_cpu[:, 1]
    w1 = state_cpu[:, 2]
    w2 = state_cpu[:, 3]
    
    # Kinetic energy
    T = 0.5 * m1 * (L1 * w1)**2 + 0.5 * m2 * ((L1 * w1)**2 + (L2 * w2)**2 + 2 * L1 * L2 * w1 * w2 * np.cos(theta1 - theta2))
    # Potential energy (relative to pivot)
    V = -m1 * g * L1 * np.cos(theta1) - m2 * g * (L1 * np.cos(theta1) + L2 * np.cos(theta2))
    return T + V

TEST_SPEC = {
    "category": "ode_conservation",
    "conserved_quantities": ["energy"],
    "also_run": ["convergence_dt"],
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the double pendulum simulation.
    """
    duration = config.get("duration", 10)
    fps = config.get("fps", 30)
    
    # Physics settings
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.81
    num_trajectories = 100
    
    # Initial conditions
    theta1_base = 2.0
    theta2_base = 2.0
    
    state = xp.zeros((num_trajectories, 4))
    state[:, 0] = theta1_base + xp.linspace(-1e-5, 1e-5, num_trajectories)
    state[:, 1] = theta2_base
    state[:, 2] = 0.0
    state[:, 3] = 0.0
    
    num_frames = int(duration * fps)
    dt_frame = 1.0 / fps
    n_substeps = 25
    dt = dt_frame / n_substeps
    
    scale = 210.0
    center_x, center_y = 540, 540
    
    trail_history = []
    max_trail_len = 80
    
    variable_logs = []
        
    def get_trail_color(i, n):
        # All trails just use COLOR_TRAIL
        return COLOR_TRAIL
    
    def get_cpu_positions(st):
        if hasattr(st, "get"):
            st_cpu = st.get()
        else:
            st_cpu = np.asarray(st)
            
        t1, t2 = st_cpu[:, 0], st_cpu[:, 1]
        x1 = L1 * np.sin(t1)
        y1 = -L1 * np.cos(t1)
        x2 = x1 + L2 * np.sin(t2)
        y2 = y1 - L2 * np.cos(t2)
        
        px1 = center_x + x1 * scale
        py1 = center_y - y1 * scale
        px2 = center_x + x2 * scale
        py2 = center_y - y2 * scale
        
        return px1, py1, px2, py2
        
    def frame_generator():
        print("Simulating double pendulum trajectories...")
        t_curr = 0.0
        
        for f in range(num_frames):
            nonlocal state
            for _ in range(n_substeps):
                state = rk4_step(double_pendulum_derivs, state, t_curr, dt, L1, L2, m1, m2, g)
                t_curr += dt
                
            # Log variables
            st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
            theta1 = st_cpu[:, 0]
            theta2 = st_cpu[:, 1]
            std_dev = float(np.std(theta2))
            
            total_E = np.mean(compute_energy(st_cpu))
            variable_logs.append([
                {"name": "Total Energy", "value": f"{total_E:.2f} J", "role": "metric", "metric_index": 0},
                {"name": "Divergence (σ)", "value": f"{std_dev:.5f}", "role": "metric", "metric_index": 1}
            ])
                
            px1, py1, px2, py2 = get_cpu_positions(state)
            
            trail_history.append((px2.copy(), py2.copy()))
            if len(trail_history) > max_trail_len:
                trail_history.pop(0)
                
            img = Image.new("RGBA", (1080, 1080), (0, 0, 0, 255))
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
                        
            # Draw physical pendulum arms
            for i in [0, num_trajectories - 1]:
                # Arm 1 (Pivot to Joint) -> Role 1 (Red)
                draw.line([(center_x, center_y), (px1[i], py1[i])], fill=COLOR_ROLE_1 + (255,), width=4)
                # Arm 2 (Joint to Tip) -> Role 2 (Blue)
                draw.line([(px1[i], py1[i]), (px2[i], py2[i])], fill=COLOR_ROLE_2 + (255,), width=4)
                
                # Draw joint and tip circles
                draw.ellipse([px1[i]-6, py1[i]-6, px1[i]+6, py1[i]+6], fill=COLOR_ROLE_1 + (255,))
                
            # Draw the pivot -> Role 3 (Green)
            draw.ellipse([center_x-8, center_y-8, center_x+8, center_y+8], fill=COLOR_ROLE_3 + (255,))
            
            # Pedagogical Annotation: Arc for theta1
            # We'll draw a dashed vertical line and an arc to the first pendulum arm
            draw.line([(center_x, center_y), (center_x, center_y + 40)], fill=(150, 150, 150, 200), width=2)
            arc_radius = 40
            t1_deg = np.degrees(theta1[0])
            # PIL arc takes [bounding box], start, end angles. 0 is +x, 90 is +y (down).
            # Vertical line is at 90 deg. The arm is at 90 - theta1 (wait, x=sin, y=cos means it's 90 - t1).
            # The angle in PIL is clockwise. Vertical is 90. The arm is 90 - t1_deg.
            start_ang, end_ang = min(90, 90 - t1_deg), max(90, 90 - t1_deg)
            draw.arc([center_x - arc_radius, center_y - arc_radius, center_x + arc_radius, center_y + arc_radius], 
                     start=start_ang, end=end_ang, fill=COLOR_ROLE_3 + (255,), width=2)
            
            # Draw the current particles at the tip
            for i in range(num_trajectories):
                draw.ellipse([px2[i]-3, py2[i]-3, px2[i]+3, py2[i]+3], fill=COLOR_ROLE_2 + (255,))
                
            yield np.array(img.convert("RGB"))
            
    return frame_generator(), variable_logs, None
