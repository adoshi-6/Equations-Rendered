import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend and integrators can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp
from integrators import rk4_step

def three_body_derivs(state, t, m1, m2, m3, G):
    """
    State: 12 variables per trajectory:
    [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]
    state shape: (num_trajectories, 12)
    """
    q1 = state[:, 0:2]
    v1 = state[:, 2:4]
    q2 = state[:, 4:6]
    v2 = state[:, 6:8]
    q3 = state[:, 8:10]
    v3 = state[:, 10:12]
    
    # Mutual distances
    d12 = xp.linalg.norm(q1 - q2, axis=1, keepdims=True)
    d23 = xp.linalg.norm(q2 - q3, axis=1, keepdims=True)
    d31 = xp.linalg.norm(q3 - q1, axis=1, keepdims=True)
    
    # Regularization to prevent division by zero
    eps = 1e-3
    d12 = xp.maximum(d12, eps)
    d23 = xp.maximum(d23, eps)
    d31 = xp.maximum(d31, eps)
    
    # Gravitational accelerations
    a1 = -G * m2 * (q1 - q2) / (d12**3) - G * m3 * (q1 - q3) / (d31**3)
    a2 = -G * m1 * (q2 - q1) / (d12**3) - G * m3 * (q2 - q3) / (d23**3)
    a3 = -G * m1 * (q3 - q1) / (d31**3) - G * m2 * (q3 - q2) / (d23**3)
    
    # Return derivatives
    deriv = xp.zeros_like(state)
    deriv[:, 0:2] = v1
    deriv[:, 2:4] = a1
    deriv[:, 4:6] = v2
    deriv[:, 6:8] = a2
    deriv[:, 8:10] = v3
    deriv[:, 10:12] = a3
    
    return deriv

# Universal Palette
COLOR_ROLE_1 = (255, 50, 50)     # Bright Red (Body 1)
COLOR_ROLE_2 = (50, 150, 255)    # Electric Blue (Body 2)
COLOR_ROLE_3 = (50, 255, 50)     # Neon Green (Body 3)
COLOR_TRAIL = (255, 120, 0)      # Orange (Paths/Trails)

from duration_utils import BoundingBoxPlateauDetector

def recommended_duration(config: dict) -> float:
    """
    Computes duration by running a coarse headless simulation to see when 
    the chaotic divergence of the three body system blooms OR reaches a spatial plateau.
    """
    G = 1.0
    m1 = m2 = m3 = 1.0
    num_trajectories = 50
    state = xp.zeros((num_trajectories, 12))
    
    x1_0, y1_0 = -0.97000436, 0.24308753
    x2_0, y2_0 = 0.97000436, -0.24308753
    x3_0, y3_0 = 0.0, 0.0
    vx1_0, vy1_0 = 0.46620531, 0.43236573
    vx2_0, vy2_0 = vx1_0, vy1_0
    vx3_0, vy3_0 = -2.0 * vx1_0, -2.0 * vy1_0
    
    state[:, 0], state[:, 1] = x1_0, y1_0
    state[:, 2], state[:, 3] = vx1_0, vy1_0
    state[:, 4], state[:, 5] = x2_0, y2_0
    state[:, 6], state[:, 7] = vx2_0, vy2_0
    state[:, 8], state[:, 9] = x3_0, y3_0
    state[:, 10], state[:, 11] = vx3_0, vy3_0
    
    # Add tiny perturbation to vx1 for chaotic divergence
    state[:, 2] += xp.linspace(-1e-4, 1e-4, num_trajectories)
    
    dt = 0.025
    max_t = 30.0
    t = 0.0
    
    detector = BoundingBoxPlateauDetector(patience_steps=50, dt=dt, rel_tolerance=1e-3)
    
    while t < max_t:
        state = rk4_step(three_body_derivs, state, t, dt, G, m1, m2, m3)
        t += dt
        
        # Calculate bounding box of all 3 bodies across all trajectories
        coords = xp.concatenate([state[:, 0:2], state[:, 4:6], state[:, 8:10]], axis=0)
        plateaued = detector.check(coords)
        
        # Measure divergence (standard deviation of body 1 x position)
        std_dev = float(xp.std(state[:, 0]))
        if plateaued or std_dev > 0.5:
            reason = "spatial extent plateaued" if plateaued else "divergence threshold (0.5) crossed"
            print(f"Three Body {reason} at {t:.1f}s. Adding 2s buffer.")
            return t + 2.0
            
    print(f"Three Body stopping criteria not met within {max_t}s.")
    return max_t

def simulate_headless(config: dict):
    duration = config.get("duration", 2.0)
    dt_divider = config.get("dt_divider", 1)
    
    G = 1.0
    m1 = m2 = m3 = 1.0
    num_trajectories = 50
    state = xp.zeros((num_trajectories, 12))
    
    x1_0, y1_0 = -0.97000436, 0.24308753
    x2_0, y2_0 = 0.97000436, -0.24308753
    x3_0, y3_0 = 0.0, 0.0
    vx1_0, vy1_0 = 0.46620531, 0.43236573
    vx2_0, vy2_0 = vx1_0, vy1_0
    vx3_0, vy3_0 = -2.0 * vx1_0, -2.0 * vy1_0
    
    state[:, 0], state[:, 1] = x1_0, y1_0
    state[:, 2], state[:, 3] = vx1_0, vy1_0
    state[:, 4], state[:, 5] = x2_0, y2_0
    state[:, 6], state[:, 7] = vx2_0, vy2_0
    state[:, 8], state[:, 9] = x3_0, y3_0
    state[:, 10], state[:, 11] = vx3_0, vy3_0
    
    state[:, 2] += xp.linspace(-1e-4, 1e-4, num_trajectories)
    
    dt = 0.025 / dt_divider
    max_t = duration
    t = 0.0
    
    states = []
    times = []
    
    while t <= max_t + 1e-5:
        states.append(state.get() if hasattr(state, "get") else np.asarray(state))
        times.append(t)
        state = rk4_step(three_body_derivs, state, t, dt, m1, m2, m3, G)
        t += dt
        
    return np.array(times), states

def compute_energy(state_cpu):
    m1 = m2 = m3 = 1.0
    G = 1.0
    q1 = state_cpu[:, 0:2]
    v1 = state_cpu[:, 2:4]
    q2 = state_cpu[:, 4:6]
    v2 = state_cpu[:, 6:8]
    q3 = state_cpu[:, 8:10]
    v3 = state_cpu[:, 10:12]
    
    T = 0.5*m1*np.sum(v1**2, axis=1) + 0.5*m2*np.sum(v2**2, axis=1) + 0.5*m3*np.sum(v3**2, axis=1)
    
    d12 = np.linalg.norm(q1 - q2, axis=1)
    d23 = np.linalg.norm(q2 - q3, axis=1)
    d31 = np.linalg.norm(q3 - q1, axis=1)
    
    V = -G*m1*m2/d12 - G*m2*m3/d23 - G*m3*m1/d31
    return T + V

def compute_momentum(state_cpu):
    m1 = m2 = m3 = 1.0
    v1 = state_cpu[:, 2:4]
    v2 = state_cpu[:, 6:8]
    v3 = state_cpu[:, 10:12]
    p = m1*v1 + m2*v2 + m3*v3
    # Return magnitude of total momentum vector
    return np.linalg.norm(p, axis=1)

TEST_SPEC = {
    "category": "ode_conservation",
    "conserved_quantities": ["energy", "momentum"],
    "also_run": ["convergence_dt"],
}

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Three-Body Problem simulation.
    """
    duration = config.get("duration", 15.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    G = 1.0
    m1 = m2 = m3 = 1.0
    num_trajectories = 50
    
    x1_0, y1_0 = -0.97000436, 0.24308753
    x2_0, y2_0 = 0.97000436, -0.24308753
    x3_0, y3_0 = 0.0, 0.0
    vx1_0, vy1_0 = 0.46620531, 0.43236573
    vx2_0, vy2_0 = vx1_0, vy1_0
    vx3_0, vy3_0 = -2.0 * vx1_0, -2.0 * vy1_0
    
    state = xp.zeros((num_trajectories, 12))
    state[:, 0], state[:, 1] = x1_0, y1_0
    state[:, 2], state[:, 3] = vx1_0, vy1_0
    state[:, 4], state[:, 5] = x2_0, y2_0
    state[:, 6], state[:, 7] = vx2_0, vy2_0
    state[:, 8], state[:, 9] = x3_0, y3_0
    state[:, 10], state[:, 11] = vx3_0, vy3_0
    
    state[:, 2] += xp.linspace(-1e-4, 1e-4, num_trajectories)
    
    dt_frame = 1.0 / fps
    n_substeps = 40
    dt = dt_frame / n_substeps
    
    width, height = 1080, 1080
    center_x, center_y = 540, 540
    scale = 320.0
    
    trail_history = []
    max_trail_len = 90
    
    frames = []
    variable_logs = []
    
    t_curr = 0.0
    for f in range(num_frames):
        for _ in range(n_substeps):
            state = rk4_step(three_body_derivs, state, t_curr, dt, m1, m2, m3, G)
            t_curr += dt
            
        st_cpu = state.get() if hasattr(state, "get") else np.asarray(state)
        
        x1, y1 = st_cpu[:, 0], st_cpu[:, 1]
        x2, y2 = st_cpu[:, 4], st_cpu[:, 5]
        x3, y3 = st_cpu[:, 8], st_cpu[:, 9]
        
        # Log divergence metric
        std_dev = float(np.std(x1))
        variable_logs.append({
            "Divergence (σ)": f"{std_dev:.4f}"
        })
        
        px1, py1 = center_x + x1 * scale, center_y - y1 * scale
        px2, py2 = center_x + x2 * scale, center_y - y2 * scale
        px3, py3 = center_x + x3 * scale, center_y - y3 * scale
        
        trail_history.append((px1.copy(), py1.copy(), px2.copy(), py2.copy(), px3.copy(), py3.copy()))
        if len(trail_history) > max_trail_len:
            trail_history.pop(0)
            
        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        
        history_len = len(trail_history)
        if history_len > 1:
            for h in range(1, history_len):
                opacity_factor = h / history_len
                alpha = int(120 * opacity_factor**1.8)
                
                p1x, p1y, p2x, p2y, p3x, p3y = trail_history[h-1]
                c1x, c1y, c2x, c2y, c3x, c3y = trail_history[h]
                
                for i in range(num_trajectories):
                    draw.line([(p1x[i], p1y[i]), (c1x[i], c1y[i])], fill=COLOR_ROLE_1 + (alpha,), width=2)
                    draw.line([(p2x[i], p2y[i]), (c2x[i], c2y[i])], fill=COLOR_ROLE_2 + (alpha,), width=2)
                    draw.line([(p3x[i], p3y[i]), (c3x[i], c3y[i])], fill=COLOR_ROLE_3 + (alpha,), width=2)
                    
        for i in range(num_trajectories):
            draw.ellipse([px1[i]-2, py1[i]-2, px1[i]+2, py1[i]+2], fill=COLOR_ROLE_1 + (255,))
            draw.ellipse([px2[i]-2, py2[i]-2, px2[i]+2, py2[i]+2], fill=COLOR_ROLE_2 + (255,))
            draw.ellipse([px3[i]-2, py3[i]-2, px3[i]+2, py3[i]+2], fill=COLOR_ROLE_3 + (255,))
            
        draw.ellipse([px1[0]-8, py1[0]-8, px1[0]+8, py1[0]+8], fill=(255, 255, 255, 255), outline=COLOR_ROLE_1 + (255,), width=2)
        draw.ellipse([px2[0]-8, py2[0]-8, px2[0]+8, py2[0]+8], fill=(255, 255, 255, 255), outline=COLOR_ROLE_2 + (255,), width=2)
        draw.ellipse([px3[0]-8, py3[0]-8, px3[0]+8, py3[0]+8], fill=(255, 255, 255, 255), outline=COLOR_ROLE_3 + (255,), width=2)
        
        frames.append(np.array(img.convert("RGB")))
        
    return frames, variable_logs
