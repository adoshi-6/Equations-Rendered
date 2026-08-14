import os
import sys
import importlib
import inspect

# Force UTF-8 stdout regardless of the OS default console codepage. Without
# this, Windows' default cp1252 console encoding cannot print the Greek
# letters used in several variable names (e.g. "Divergence (σ)", used by
# lorenz/rossler/three_body/double_pendulum's trend_assertions output),
# crashing the entire test run with a UnicodeEncodeError partway through —
# confirmed as a real failure on Windows during verification, not a
# hypothetical. This makes the test runner's output encoding-safe on any
# platform rather than only ever having been exercised on a UTF-8-default
# Linux environment.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_framework

def run_category(category, module, config, spec):
    """
    Runs a single TEST_SPEC category, prints the same human-readable output
    as before, and ALSO returns a structured result dict:
    {"category": str, "passed": bool, "checks": [...], "error": str|None}

    This was refactored from a print-only function so render_service.py's
    /verify endpoint can get a real, structured pass/fail signal directly
    (by importing and calling this function) instead of having to spawn a
    subprocess and fragile-parse "[PASS]"/"[FAIL]" text out of stdout. The
    printed output is unchanged — same strings, same formatting — so
    anyone running `python run_physics_tests.py` from the CLI sees exactly
    what they saw before this refactor.
    """
    print(f" > Test Category: {category}")
    checks = []
    overall_passed = True
    error = None
    try:
        if category == "ode_conservation":
            res = physics_framework.test_conservation(module, config, spec["conserved_quantities"])
            for qty, data in res.items():
                status = "PASS" if data["passed"] else "FAIL"
                print(f"   [{status}] Conservation of {qty}")
                print(f"          Start: {data['start']:.6f} | End: {data['end']:.6f} | Drift: {data['drift']:.6f}")
                checks.append({"description": f"Conservation of {qty}", "passed": data["passed"], "detail": data})
                overall_passed = overall_passed and data["passed"]

        elif category == "bounded_region":
            res = physics_framework.test_bounded_region(module, config, spec["expected_bounds"])
            for var, data in res.items():
                status = "PASS" if data["passed"] else "FAIL"
                print(f"   [{status}] Variable {var}: min={data['actual_min']:.4f}, max={data['actual_max']:.4f}")
                print(f"          Expected bounds: [{data['expected_min']}, {data['expected_max']}]")
                checks.append({"description": f"Bounded region: {var}", "passed": data["passed"], "detail": data})
                overall_passed = overall_passed and data["passed"]

        elif category == "convergence_dt":
            res = physics_framework.test_convergence_dt(module, config)
            status = "PASS" if res["passed"] else "FAIL"
            print(f"   [{status}] max difference when dt -> dt/2: {res['max_difference']:.8f}")
            checks.append({"description": "convergence_dt", "passed": res["passed"], "detail": res})
            overall_passed = overall_passed and res["passed"]

        elif category == "known_points":
            res = physics_framework.test_known_points(module, config, spec["known_points"])
            for r in res:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"   [{status}] Input {r['input']}")
                print(f"          Expected: {r['expected']} | Actual: {r['actual']}")
                checks.append({"description": f"known_point: {r['input']}", "passed": r["passed"], "detail": r})
                overall_passed = overall_passed and r["passed"]

        elif category == "bifurcation":
            res = physics_framework.test_bifurcation(module, config, spec["known_transitions"])
            for tr, data in res.items():
                status = "PASS" if data["passed"] else "FAIL"
                act = f"{data['actual']:.4f}" if data['actual'] is not None else "None"
                print(f"   [{status}] Transition '{tr}': expected r={data['expected']:.4f}, actual r={act}")
                checks.append({"description": f"bifurcation: {tr}", "passed": data["passed"], "detail": data})
                overall_passed = overall_passed and data["passed"]

        elif category == "optimization_convergence":
            res = physics_framework.test_optimization_convergence(module, config, spec["known_minima"], spec["tolerance"])
            status = "PASS" if res["passed"] else "FAIL"
            print(f"   [{status}] Final Position: ({res['final_position'][0]:.4f}, {res['final_position'][1]:.4f})")
            if res['nearest_minimum']:
                print(f"          Nearest Known Minimum: {res['nearest_minimum']}")
                print(f"          Distance: {res['distance']:.6f} (Tolerance: {res['tolerance']})")
            checks.append({"description": "optimization_convergence", "passed": res["passed"], "detail": res})
            overall_passed = overall_passed and res["passed"]

        elif category == "ensemble_stats":
            tol = spec.get("tolerance_percent", 10.0)
            res = physics_framework.test_ensemble_stats(module, config, spec["expected_stats"], tol)
            for stat, data in res["evaluations"].items():
                status = "PASS" if data["passed"] else "FAIL"
                print(f"   [{status}] {stat}: expected {data['expected']:.4f}, actual {data['actual']:.4f}")
                print(f"          Difference: {data['diff_pct']:.2f}% (Tolerance: {res['tolerance_pct']}%)")
                checks.append({"description": f"ensemble_stats: {stat}", "passed": data["passed"], "detail": data})
                overall_passed = overall_passed and data["passed"]

        elif category == "trend_assertions":
            res = physics_framework.test_trend_assertions(module, config, spec["trend_assertions"])
            if "error" in res:
                print(f"   [ERROR] {res['error']}")
                error = res["error"]
                overall_passed = False
            else:
                for var, data in res.items():
                    if var == "overall_passed":
                        continue
                    status = "PASS" if data.get("passed") else "FAIL"
                    if "error" in data:
                        print(f"   [{status}] {var}: {data['error']}")
                    else:
                        print(f"   [{status}] {var} ({data['expected']}): v1={data['v1']:.4f}, v2={data['v2']:.4f}, v3={data['v3']:.4f}")
                    checks.append({"description": f"trend_assertions: {var}", "passed": data.get("passed", False), "detail": data})
                    overall_passed = overall_passed and data.get("passed", False)

        else:
            print(f"   [WARNING] Unknown category: {category}")
            error = f"Unknown category: {category}"
            overall_passed = False

    except Exception as e:
        import traceback
        print(f"   [ERROR] Failed running {category}: {e}")
        traceback.print_exc()
        error = str(e)
        overall_passed = False

    return {"category": category, "passed": overall_passed, "checks": checks, "error": error}


def run_tests_for_module(module_name: str) -> dict:
    """
    Runs every TEST_SPEC category declared for a single named simulation
    module and returns one fully structured result:
    {"simulation": str, "passed": bool, "has_test_spec": bool, "categories": [...]}

    Built specifically so render_service.py's /verify endpoint can call this
    directly (import + function call) rather than spawning a subprocess and
    parsing stdout text — a real, structured pass/fail signal instead of a
    fragile text-scraping layer.
    """
    simulations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations")
    try:
        module = importlib.import_module(f"simulations.{module_name}")
    except ImportError as e:
        return {"simulation": module_name, "passed": False, "has_test_spec": False,
                "categories": [], "error": f"Could not import {module_name}: {e}"}

    if not hasattr(module, "TEST_SPEC"):
        return {"simulation": module_name, "passed": True, "has_test_spec": False,
                "categories": [], "error": None}

    spec = module.TEST_SPEC
    config = {}
    config_path = os.path.join(os.path.dirname(simulations_dir), "configs", f"{module_name}.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif hasattr(module, "recommended_duration"):
        config["duration"] = module.recommended_duration(config)

    print(f"--- Running tests for: {module_name} ---")
    categories = []
    overall_passed = True

    category = spec.get("category")
    if category:
        result = run_category(category, module, config, spec)
        categories.append(result)
        overall_passed = overall_passed and result["passed"]

    for extra in spec.get("also_run", []):
        result = run_category(extra, module, config, spec)
        categories.append(result)
        overall_passed = overall_passed and result["passed"]

    print("\n")
    return {"simulation": module_name, "passed": overall_passed, "has_test_spec": True,
            "categories": categories, "error": None}


def run_tests():
    simulations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations")

    print("========================================")
    print("   PHYSICS & MATH VERIFICATION RUNNER   ")
    print("========================================\n")

    for filename in sorted(os.listdir(simulations_dir)):
        if filename.endswith(".py") and filename != "dummy.py":
            module_name = filename[:-3]
            run_tests_for_module(module_name)

if __name__ == "__main__":
    run_tests()
