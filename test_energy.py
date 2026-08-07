import sys
sys.path.append('.')
import numpy as np
from simulations.double_pendulum import rk4_step, double_pendulum_derivs

def compute_energy(state, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    th1, th2, w1, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    v1 = L1**2 * m1 * w1**2 / 2.0
    v2 = m2 / 2.0 * (L1**2 * w1**2 + L2**2 * w2**2 + 2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2))
    T = v1 + v2
    y1 = -L1 * np.cos(th1)
    y2 = y1 - L2 * np.cos(th2)
    V = m1 * g * y1 + m2 * g * y2
    return T + V

state = np.zeros((1, 4))
state[0, 0] = 2.0
state[0, 1] = 2.0
L1, L2, m1, m2, g = 1.0, 1.0, 1.0, 1.0, 9.81
dt = 1.0 / (30 * 25)

e0 = compute_energy(state)[0]
t = 0.0
for _ in range(30 * 10 * 25):  # 10 seconds at 30 fps * 25 substeps
    state = rk4_step(double_pendulum_derivs, state, t, dt, L1, L2, m1, m2, g)
    t += dt

e_end = compute_energy(state)[0]
print(f"Energy start: {e0:.4f}")
print(f"Energy end: {e_end:.4f}")
print(f"Drift: {e_end - e0:.4f}")
