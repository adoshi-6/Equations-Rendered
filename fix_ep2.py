with open('simulations/epicycloid.py', 'r') as f:
    content = f.read()

target = "    max_t = 2 * np.pi\n    t_vals = np.linspace(0, max_t, num_frames)"

if target in content:
    # Remove it from where it is
    content = content.replace(target, "")
    # Add it before the bounds check
    bounds_check = "# PRECOMPUTE BOUNDS (Section 3.6)"
    content = content.replace(bounds_check, target + "\n    " + bounds_check)
    with open('simulations/epicycloid.py', 'w') as f:
        f.write(content)
    print("Fixed epicycloid.py t_vals order")
else:
    print("Target not found")
