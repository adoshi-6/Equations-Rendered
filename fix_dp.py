import re

with open('simulations/double_pendulum.py', 'r') as f:
    content = f.read()

pre_sim_code = """
    # PRECOMPUTE BOUNDS (Section 3.6)
    import numpy as np
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    t_curr_pre = 0.0
    temp_state = xp.zeros((num_trajectories, 4))
    temp_state[:, 0] = theta1_base + xp.linspace(-1e-5, 1e-5, num_trajectories)
    temp_state[:, 1] = theta2_base
    temp_state[:, 2] = 0.0
    temp_state[:, 3] = 0.0
    
    for f in range(num_frames):
        for _ in range(n_substeps):
            temp_state = rk4_step(double_pendulum_derivs, temp_state, t_curr_pre, dt, L1, L2, m1, m2, g)
            t_curr_pre += dt
            
        st_cpu = temp_state.get() if hasattr(temp_state, "get") else np.asarray(temp_state)
        t1, t2 = st_cpu[:, 0], st_cpu[:, 1]
        x1_pre = L1 * np.sin(t1)
        y1_pre = -L1 * np.cos(t1)
        x2_pre = x1_pre + L2 * np.sin(t2)
        y2_pre = y1_pre - L2 * np.cos(t2)
        
        min_x = min(min_x, float(np.min(x1_pre)), float(np.min(x2_pre)), 0.0)
        max_x = max(max_x, float(np.max(x1_pre)), float(np.max(x2_pre)), 0.0)
        min_y = min(min_y, float(np.min(y1_pre)), float(np.min(y2_pre)), 0.0)
        max_y = max(max_y, float(np.max(y1_pre)), float(np.max(y2_pre)), 0.0)
        
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
"""

target = """    scale = 210.0
    center_x, center_y = 540, 540"""

if target in content:
    new_content = content.replace(target, pre_sim_code)
    with open('simulations/double_pendulum.py', 'w') as f:
        f.write(new_content)
    print("Fixed double_pendulum.py")
else:
    print("Could not find target in double_pendulum.py")
