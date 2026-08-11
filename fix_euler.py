with open('simulations/euler.py', 'r') as f:
    content = f.read()

pre_sim_code = """
    # PRECOMPUTE BOUNDS (Section 3.6)
    # Trajectory is unit circle: x in [-1, 1], y in [-1, 1].
    # Annotations: axes go from -1.333 to +1.333.
    # Hatching is at x = -1.333.
    min_x, max_x = -1.333, 1.333
    min_y, max_y = -1.333, 1.333
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    margin_x = range_x * 0.10
    margin_y = range_y * 0.10
    
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    
    scale_x = 1080 / (max_x - min_x)
    scale_y = 1080 / (max_y - min_y)
    radius = float(min(scale_x, scale_y))
    
    # In Euler, cx and cy are hardcoded. We compute them dynamically:
    cx = int(1080 / 2 - ((min_x + max_x) / 2) * radius)
    cy = int(1080 / 2 + ((min_y + max_y) / 2) * radius)
"""

target = """    cx, cy = w // 2, h // 2
    radius = 300"""

if target in content:
    new_content = content.replace(target, pre_sim_code)
    
    # We must also replace the hardcoded "400" in euler with "1.333 * radius"
    new_content = new_content.replace('400', 'int(1.333 * radius)')
    
    with open('simulations/euler.py', 'w') as f:
        f.write(new_content)
    print("Fixed euler.py")
else:
    print("Could not find target in euler.py")
