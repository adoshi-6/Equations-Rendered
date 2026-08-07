import os
import subprocess

sims = ['rossler', 'julia', 'epicycloid', 'random_walk', 'standing_wave']

for sim in sims:
    config_path = f'configs/{sim}.yaml'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(f'module: "{sim}"\nduration: 15.0\nfps: 30\n')

    print(f'Rendering {sim}...')
    subprocess.run([
        'C:\\Users\\Aryan\\AppData\\Local\\Python\\bin\\python.exe', 
        'renderer.py', 
        '--config', config_path,
        '--output', f'C:\\Users\\Aryan\\.gemini\\antigravity\\brain\\02ecd8e5-f0ac-4e84-a4bb-e91012038d0a\\{sim}.mp4'
    ], check=True)
