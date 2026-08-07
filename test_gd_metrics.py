import sys
sys.path.append('.')
import numpy as xp

def himmelblau(x, y):
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7.0)**2

def himmelblau_grad(x, y):
    dfdx = 4.0 * x * (x**2 + y - 11.0) + 2.0 * (x + y**2 - 7.0)
    dfdy = 2.0 * (x**2 + y - 11.0) + 4.0 * y * (x + y**2 - 7.0)
    return dfdx, dfdy

import numpy as np
np.random.seed(42)
px = xp.asarray(np.random.uniform(-5.0, 5.0, 120))
py = xp.asarray(np.random.uniform(-5.0, 5.0, 120))

gx, gy = himmelblau_grad(px, py)
mag = xp.sqrt(gx**2 + gy**2) + 1e-8
mean_loss = float(xp.mean(himmelblau(px, py)))
mean_grad = float(xp.mean(mag))

print(f"Avg Loss: {mean_loss:.2f}")
print(f"Avg Gradient: {mean_grad:.2f}")
