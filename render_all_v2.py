import os
import subprocess

sims = ['rossler', 'julia', 'epicycloid', 'random_walk', 'standing_wave']
artifact_dir = r"C:\Users\Aryan\.gemini\antigravity\brain\02ecd8e5-f0ac-4e84-a4bb-e91012038d0a"

equations = {
    'rossler': r"dx/dt = -y-z, \quad dy/dt = x+ay, \quad dz/dt = b+z(x-c)",
    'julia': r"z_{n+1} = z_n^2 + c",
    'epicycloid': r"x(t) = R(k+1)\cos(t) - R\cos((k+1)t), \quad y(t) = R(k+1)\sin(t) - R\sin((k+1)t)",
    'random_walk': r"dx = \sqrt{2D}\, dW_x, \quad dy = \sqrt{2D}\, dW_y",
    'standing_wave': r"y(x,t) = \sin(kx - \omega t) + \sin(kx + \omega t)"
}

for sim in sims:
    config_path = f'configs/{sim}.yaml'
    # Always recreate to ensure it has the right keys
    title = sim.replace('_', ' ').title()
    eq = equations[sim].replace('\\', '\\\\')
    with open(config_path, 'w') as f:
        f.write(f'simulation: "{sim}"\ntitle: "{title}"\nequation: "{eq}"\nduration: 15.0\nfps: 30\n')

    print(f'Rendering {sim}...')
    subprocess.run([
        r'C:\Users\Aryan\AppData\Local\Python\bin\python.exe', 
        'renderer.py', 
        '--config', config_path,
        '--output', os.path.join(artifact_dir, f'{sim}.mp4')
    ], check=True)
