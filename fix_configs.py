import os, glob, yaml, subprocess

configs = glob.glob('configs/*.yaml')
for c in configs:
    with open(c, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    if cfg and 'duration' in cfg:
        print(f"Removing hardcoded duration from {c} (was {cfg['duration']})")
        del cfg['duration']
        with open(c, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False)

# Render julia since its auto-computed duration is 20s, but it was forced to 15s before
print('Rendering julia (now that duration is auto-computed)...')
artifact_dir = r'C:\Users\Aryan\.gemini\antigravity\brain\02ecd8e5-f0ac-4e84-a4bb-e91012038d0a'
subprocess.run([
    r'C:\Users\Aryan\AppData\Local\Python\bin\python.exe', 
    'renderer.py', 
    '--config', 'configs/julia.yaml',
    '--output', os.path.join(artifact_dir, 'julia.mp4')
], check=True)

print('Extracting frames...')
subprocess.run([
    r'C:\Users\Aryan\AppData\Local\Python\bin\python.exe', 
    'extract_frames.py'
], check=True)
