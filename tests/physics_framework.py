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

def _extract_var_value(log_entry, var_name):
    """
    Extracts a numeric value for var_name from a single variable_log entry,
    supporting BOTH schemas currently in use across the codebase:
      - legacy flat dict:      {"Divergence (\u03c3)": "8.93312", ...}
      - current list-of-dicts: [{"name": "Divergence (\u03c3)", "value": "8.93312", ...}, ...]
    Returns None if var_name isn't present in this entry.
    (This function previously only handled the legacy dict format via
    `if var_name in log: log[var_name]`, which silently found nothing for
    every simulation migrated to the list-of-dicts schema — a confirmed real
    bug that made trend-assertion testing non-functional for every migrated
    simulation. Fixed here to support both.)
    """
    if isinstance(log_entry, dict):
        if var_name in log_entry:
            return log_entry[var_name]
        return None
    if isinstance(log_entry, list):
        for item in log_entry:
            if isinstance(item, dict) and item.get("name") == var_name:
                return item.get("value")
        return None
    return None


def test_trend_assertions(module, config, assertions):
    """
    Verifies that certain metrics trend in expected directions over the course of the simulation.
    E.g. monotonic_increase, monotonic_decrease
    """
    try:
        # Use generate() to get variables logs since simulate_headless might not return them
        sim_output = module.generate(config)
        if isinstance(sim_output, tuple):
            if len(sim_output) == 4:
                frame_gen, variable_logs, _, _ = sim_output
            elif len(sim_output) == 3:
                frame_gen, variable_logs, _ = sim_output
            elif len(sim_output) == 2:
                frame_gen, variable_logs = sim_output
            else:
                return {"error": f"Module generate() returned unexpected tuple length {len(sim_output)}", "passed": False}
        else:
            return {"error": "Module generate() did not return variable_logs", "passed": False}

        # CRITICAL: variable_logs is populated incrementally, inside the frame
        # generator's body, for every correctly-implemented simulation (this
        # is the architecturally-required pattern — see renderer.py's
        # variable_logs[-1] usage). That means variable_logs stays EMPTY until
        # the generator is actually driven forward. Previously this function
        # never iterated frame_gen at all, so every trend assertion silently
        # failed with "not found" regardless of the simulation or schema —
        # not a data problem, a "the generator was never run" problem. Drain
        # it here (frames themselves aren't needed for this test).
        for _ in frame_gen:
            pass
    except Exception as e:
        return {"error": str(e), "passed": False}
        
    results = {}
    overall_passed = True
    
    for var_name, expected_trend in assertions.items():
        var_values = []
        for log in variable_logs:
            raw_val = _extract_var_value(log, var_name)
            if raw_val is None:
                continue
            try:
                # Values are often formatted strings like "78.93" or "12.25 J" —
                # strip any trailing non-numeric unit/suffix before parsing.
                import re as _re
                m = _re.match(r'^\s*(-?\d+\.?\d*)', str(raw_val))
                if m:
                    var_values.append(float(m.group(1)))
            except (ValueError, TypeError):
                pass
                    
        if not var_values:
            results[var_name] = {"passed": False, "error": f"Variable '{var_name}' not found in logs"}
            overall_passed = False
            continue
            
        passed = True
        
        # sample at 10%, 50%, 90%
        idx1 = max(0, int(len(var_values) * 0.1))
        idx2 = max(0, int(len(var_values) * 0.5))
        idx3 = max(0, int(len(var_values) * 0.9))
        
        v1, v2, v3 = var_values[idx1], var_values[idx2], var_values[idx3]
        
        if expected_trend == "monotonic_increase":
            if not (v1 <= v2 <= v3) or (v1 == v3 and v1 != 0): # Needs to grow, not stay flat (unless 0)
                passed = False
        elif expected_trend == "monotonic_decrease":
            if not (v1 >= v2 >= v3):
                passed = False
                
        results[var_name] = {
            "expected": expected_trend,
            "v1": v1, "v2": v2, "v3": v3,
            "passed": passed
        }
        if not passed:
            overall_passed = False
            
    results["overall_passed"] = overall_passed
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
