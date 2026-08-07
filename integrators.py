from backend import xp

def rk4_step(derivs, state, t, dt, *args, **kwargs):
    """
    Computes a single Runge-Kutta 4th order step for a batch of states.
    
    Parameters:
    - derivs: Callable function derivs(state, t, *args, **kwargs) returning the derivatives.
              Must return an array of the same shape as state.
    - state: Array of shape (num_trajectories, state_dim) or (state_dim,) using backend.xp.
    - t: Current time (scalar).
    - dt: Time step size (scalar).
    
    Returns:
    - Next state of the same shape.
    """
    k1 = derivs(state, t, *args, **kwargs)
    k2 = derivs(state + 0.5 * dt * k1, t + 0.5 * dt, *args, **kwargs)
    k3 = derivs(state + 0.5 * dt * k2, t + 0.5 * dt, *args, **kwargs)
    k4 = derivs(state + dt * k3, t + dt, *args, **kwargs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
