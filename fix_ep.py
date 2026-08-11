with open('simulations/epicycloid.py', 'r') as f:
    content = f.read()

pre_sim_code = """
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
"""

target = """    center_x, center_y = 540, 540
    scale = 100.0"""

if target in content:
    new_content = content.replace(target, pre_sim_code)
    with open('simulations/epicycloid.py', 'w') as f:
        f.write(new_content)
    print("Fixed epicycloid.py")
else:
    print("Could not find target in epicycloid.py")
