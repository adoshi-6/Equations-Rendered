import os
import glob
import subprocess
import time
import sys

def main():
    configs_dir = "configs"
    output_dir = "output"
    baseline_dir = "tests/baseline_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)
    
    yaml_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    yaml_files.sort()
    
    print(f"Found {len(yaml_files)} simulations to render.")
    
    for yaml_file in yaml_files:
        basename = os.path.basename(yaml_file)
        sim_name = basename[:-5]
        
        # skip dummy
        if sim_name == "dummy":
            continue
            
        output_path = os.path.join(output_dir, f"{sim_name}.mp4")
        
        print(f"\n=========================================")
        print(f" Rendering {sim_name} ...")
        print(f"=========================================")
        
        start_time = time.time()
        
        cmd = [
            sys.executable, "renderer.py",
            "--config", yaml_file,
            "--output", output_path,
            "--baseline-dir", baseline_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            print(f"Successfully rendered {sim_name} in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"Failed to render {sim_name}. Error: {e}")
            
if __name__ == "__main__":
    main()
