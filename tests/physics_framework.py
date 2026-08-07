import numpy as np

def test_conservation(module, config, conserved_quantities, tolerance=0.1):
    t_array, states = module.simulate_headless(config)
    results = {}
    for qty in conserved_quantities:
        func = getattr(module, f"compute_{qty}")
        val_start = float(np.mean(func(states[0])))
        val_end = float(np.mean(func(states[-1])))
        drift = val_end - val_start
        passed = abs(drift) <= tolerance
        results[qty] = {"start": val_start, "end": val_end, "drift": drift, "passed": passed}
    return results

def test_bounded_region(module, config, expected_bounds):
    t_array, states = module.simulate_headless(config)
    results = {}
    variables = module.get_state_variables(states)
    for var, bounds in expected_bounds.items():
        actual_min = float(np.min(variables[var]))
        actual_max = float(np.max(variables[var]))
        expected_min = bounds.get("min", -np.inf)
        expected_max = bounds.get("max", np.inf)
        passed = (actual_min >= expected_min) and (actual_max <= expected_max)
        results[var] = {
            "actual_min": actual_min, "actual_max": actual_max,
            "expected_min": expected_min, "expected_max": expected_max,
            "passed": passed
        }
    return results

def test_convergence_dt(module, config):
    config_1 = config.copy()
    config_1["duration"] = 1.0  # Run a short simulation before chaos
    config_2 = config_1.copy()
    config_2["dt_divider"] = 2
    
    t1, states1 = module.simulate_headless(config_1)
    t2, states2 = module.simulate_headless(config_2)
    
    # Compare final states
    diff = float(np.max(np.abs(states1[-1] - states2[-1])))
    passed = diff < 1e-2
    return {"max_difference": diff, "passed": passed}

def test_known_points(module, config, known_points):
    results = []
    for inp, expected, tol in known_points:
        actual = module.evaluate_point(inp)
        passed = True
        diffs = {}
        for k, v in expected.items():
            if tol is not None:
                diff = abs(actual[k] - v)
                diffs[k] = diff
                if diff > tol: passed = False
            else:
                diffs[k] = actual[k] == v
                if actual[k] != v: passed = False
        results.append({"input": inp, "expected": expected, "actual": actual, "passed": passed, "diffs": diffs})
    return results

def test_bifurcation(module, config, known_transitions):
    actual_transitions = module.find_bifurcation_transitions(config)
    results = {}
    for name, exp_r in known_transitions.items():
        act_r = actual_transitions.get(name)
        if act_r is not None:
            diff = abs(act_r - exp_r)
            results[name] = {"expected": exp_r, "actual": act_r, "diff": diff, "passed": diff < 0.05}
        else:
            results[name] = {"expected": exp_r, "actual": None, "diff": None, "passed": False}
    return results

def test_optimization_convergence(module, config, known_minima, tolerance):
    final_pos = module.run_optimization(config)
    
    min_dist = float('inf')
    best_min = None
    for km in known_minima:
        dist = np.sqrt((final_pos[0] - km[0])**2 + (final_pos[1] - km[1])**2)
        if dist < min_dist:
            min_dist = dist
            best_min = km
            
    passed = min_dist <= tolerance
    return {
        "final_position": final_pos,
        "nearest_minimum": best_min,
        "distance": float(min_dist),
        "tolerance": tolerance,
        "passed": passed
    }

def test_ensemble_stats(module, config, expected_stats, tolerance_percent=10.0):
    results = module.get_ensemble_stats(config)
    passed = True
    evaluations = {}
    for stat, expected_val in expected_stats.items():
        actual_val = results.get(stat)
        if actual_val is None:
            passed = False
            continue
        
        diff_pct = abs(actual_val - expected_val) / expected_val * 100.0 if expected_val != 0 else 0
        stat_passed = diff_pct <= tolerance_percent
        if not stat_passed:
            passed = False
        evaluations[stat] = {
            'expected': expected_val,
            'actual': actual_val,
            'diff_pct': diff_pct,
            'passed': stat_passed
        }
    return {'evaluations': evaluations, 'passed': passed, 'tolerance_pct': tolerance_percent}
