import os
import glob

sim_files = glob.glob('simulations/*.py') + ['renderer.py']
for fpath in sim_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('(10, 10, 15)', '(0, 0, 0)')
    new_content = new_content.replace('#0a0a0f', '#000000')
    
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {fpath}')
print('Done!')
