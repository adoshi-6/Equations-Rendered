import os
import sys
import importlib
import inspect

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_framework

def run_tests():
    simulations_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations")
    
    print("========================================")
    print("   PHYSICS & MATH VERIFICATION RUNNER   ")
    print("========================================\n")
    
    for filename in sorted(os.listdir(simulations_dir)):
        if filename.endswith(".py") and filename != "dummy.py":
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"simulations.{module_name}")
            except ImportError as e:
                print(f"[ERROR] Could not import {module_name}: {e}")
                continue
                
            if not hasattr(module, "TEST_SPEC"):
                continue
                
            spec = module.TEST_SPEC
            print(f"--- Running tests for: {module_name} ---")
            
            config = {}  # Base config, can be overridden by specific tests if needed
            config_path = os.path.join(os.path.dirname(simulations_dir), "configs", f"{module_name}.yaml")
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                # If no config, try calling recommended_duration to get the correct duration
                if hasattr(module, "recommended_duration"):
                    config["duration"] = module.recommended_duration(config)
            
            # Primary test category
            category = spec.get("category")
            if category:
                run_category(category, module, config, spec)
                
            # Additional tests
            for extra in spec.get("also_run", []):
                run_category(extra, module, config, spec)
                
            print("\n")

def run_category(category, module, config, spec):
    print(f" > Test Category: {category}")
    try:
        if category == "ode_conservation":
            res = physics_framework.test_conservation(module, config, spec["conserved_quantities"])
            for qty, data in res.items():
                status = "PASS" if data["passed"] else "FAIL"
                print(f"   [{status}] Conservation of {qty}")
                print(f"          Start: {data['start']:.6f} | End: {data['end']:.6f} | Drift: {data['drift']:.6f}")
                
        elif category == "bounded_region":
            res = physics_framework.test_bounded_region(module, config, spec["expected_bounds"])
            for var, data in res.items():
                status = "PASS" if data["passed"] else "FAIL"
                print(f"   [{status}] Variable {var}: min={data['actual_min']:.4f}, max={data['actual_max']:.4f}")
                print(f"          Expected bounds: [{data['expected_min']}, {data['expected_max']}]")
                
        elif category == "convergence_dt":
            res = physics_framework.test_convergence_dt(module, config)
            status = "PASS" if res["passed"] else "FAIL"
            print(f"   [{status}] max difference when dt -> dt/2: {res['max_difference']:.8f}")
            
        elif category == "known_points":
            res = physics_framework.test_known_points(module, config, spec["known_points"])
            for r in res:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"   [{status}] Input {r['input']}")
                print(f"          Expected: {r['expected']} | Actual: {r['actual']}")
                
        elif category == "bifurcation":
            res = physics_framework.test_bifurcation(module, config, spec["known_transitions"])
            for tr, data in res.items():
                status = "PASS" if data["passed"] else "FAIL"
                act = f"{data['actual']:.4f}" if data['actual'] is not None else "None"
                print(f"   [{status}] Transition '{tr}': expected r={data['expected']:.4f}, actual r={act}")
                
        elif category == "optimization_convergence":
            res = physics_framework.test_optimization_convergence(module, config, spec["known_minima"], spec["tolerance"])
            status = "PASS" if res["passed"] else "FAIL"
            print(f"   [{status}] Final Position: ({res['final_position'][0]:.4f}, {res['final_position'][1]:.4f})")
            if res['nearest_minimum']:
                print(f"          Nearest Known Minimum: {res['nearest_minimum']}")
                print(f"          Distance: {res['distance']:.6f} (Tolerance: {res['tolerance']})")
                
        elif category == "ensemble_stats":
            tol = spec.get("tolerance_percent", 10.0)
            res = physics_framework.test_ensemble_stats(module, config, spec["expected_stats"], tol)
            for stat, data in res["evaluations"].items():
                status = "PASS" if data["passed"] else "FAIL"
                print(f"   [{status}] {stat}: expected {data['expected']:.4f}, actual {data['actual']:.4f}")
                print(f"          Difference: {data['diff_pct']:.2f}% (Tolerance: {res['tolerance_pct']}%)")

        elif category == "trend_assertions":
            # NOTE: this category was previously completely unreachable — no
            # dispatch branch existed for it at all, so test_trend_assertions()
            # never actually ran for any simulation regardless of TEST_SPEC
            # contents. Wired in here.
            res = physics_framework.test_trend_assertions(module, config, spec["trend_assertions"])
            if "error" in res:
                print(f"   [ERROR] {res['error']}")
            else:
                for var, data in res.items():
                    if var == "overall_passed":
                        continue
                    status = "PASS" if data.get("passed") else "FAIL"
                    if "error" in data:
                        print(f"   [{status}] {var}: {data['error']}")
                    else:
                        print(f"   [{status}] {var} ({data['expected']}): v1={data['v1']:.4f}, v2={data['v2']:.4f}, v3={data['v3']:.4f}")

        else:
            print(f"   [WARNING] Unknown category: {category}")
            
    except Exception as e:
        import traceback
        print(f"   [ERROR] Failed running {category}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
