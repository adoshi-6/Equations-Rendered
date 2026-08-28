import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

def himmelblau(x, y):
    """Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2"""
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7.0)**2

def himmelblau_grad(x, y):
    """Gradient of Himmelblau's function."""
    dfdx = 4.0 * x * (x**2 + y - 11.0) + 2.0 * (x + y**2 - 7.0)
    dfdy = 2.0 * (x**2 + y - 11.0) + 4.0 * y * (x + y**2 - 7.0)
    return dfdx, dfdy

# Universal Palette (Priority 5 - Desaturated to match ROLE_COLORS)
COLOR_ROLE_1 = (232, 93, 74)     # Primary (#E85D4A)
COLOR_ROLE_2 = (93, 168, 232)    # Secondary (#5DA8E8)
COLOR_ROLE_3 = (127, 174, 107)   # Auxiliary (#7FAE6B)
COLOR_TRAIL = (232, 93, 74)      # Match primary for trails if used

def run_optimization(config):
    # run gradient descent on a single point to find minimum
    px, py = 0.0, 0.0
    learning_rate = 0.004
    for _ in range(1000):
        gx, gy = himmelblau_grad(px, py)
        px -= learning_rate * gx
        py -= learning_rate * gy
    return (float(px), float(py))

TEST_SPEC = {
    "category": "optimization_convergence",
    "known_minima": [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126)
    ],
    "tolerance": 0.05,
    "also_run": ["trend_assertions"],
    # Matches the empirically-verified real headless trace from manual review
    # (Avg Loss: 138.25 -> 83.54 -> 28.22 -> 6.58 -> 0.55 -> 0.00). Now an
    # automated regression check instead of something that has to be manually
    # re-verified by eye every time.
    "trend_assertions": {
        "Avg Loss": "monotonic_decrease"
    }
}

def recommended_duration(config: dict) -> float:
    """
    Computes duration by running a headless simulation to see when 
    the particles converge to the minima.
    """
    num_particles = 120
    learning_rate = 0.004
    x_min, x_max = -5.5, 5.5
    y_min, y_max = -5.5, 5.5
    
    np.random.seed(42)
    px = xp.asarray(np.random.uniform(x_min + 0.5, x_max - 0.5, num_particles))
    py = xp.asarray(np.random.uniform(y_min + 0.5, y_max - 0.5, num_particles))
    
    dt = 1.0 / 30.0
    max_t = 30.0
    t = 0.0
    
    while t < max_t:
        gx, gy = himmelblau_grad(px, py)
        mag = xp.sqrt(gx**2 + gy**2)
        mean_grad = float(xp.mean(mag))
        
        if mean_grad < 0.5:
            print(f"Gradient Descent converged at {t:.1f}s.")
            return t
            
        scale = xp.minimum(mag + 1e-8, 5.0) / (mag + 1e-8)
        px = px - learning_rate * gx * scale
        py = py - learning_rate * gy * scale
        
        t += dt
        
    return max_t

def particle_color(i, n):
    """Assigns universal palette colors to particles."""
    t = i / max(1, n - 1)
    if t < 0.33:
        return COLOR_ROLE_1
    elif t < 0.66:
        return COLOR_ROLE_2
    else:
        return COLOR_ROLE_3

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Gradient Descent on Himmelblau's function.
    """
    duration = config.get("duration", 12.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)

    width, height = 1080, 1080

    x_min, x_max = -5.5, 5.5
    y_min, y_max = -5.5, 5.5

    u = xp.linspace(x_min, x_max, width)
    v = xp.linspace(y_max, y_min, height)
    uu, vv = xp.meshgrid(u, v)
    z = himmelblau(uu, vv)

    z_log = xp.log1p(z)
    z_max = xp.max(z_log)
    z_norm = z_log / z_max

    bg_r = xp.clip(30.0 + 40.0 * xp.sin(xp.pi * z_norm * 6.0) * (1.0 - z_norm), 0, 255)
    bg_g = xp.clip(15.0 + 60.0 * xp.sin(xp.pi * z_norm * 4.0) * (1.0 - z_norm), 0, 255)
    bg_b = xp.clip(50.0 + 80.0 * xp.sin(xp.pi * z_norm * 3.0) * (1.0 - z_norm), 0, 255)

    bg = xp.stack([bg_r, bg_g, bg_b], axis=-1).astype(xp.uint8)
    if hasattr(bg, "get"):
        bg_cpu = bg.get()
    else:
        bg_cpu = np.asarray(bg)

    num_particles = 120
    learning_rate = 0.004

    np.random.seed(42)
    px = xp.asarray(np.random.uniform(x_min + 0.5, x_max - 0.5, num_particles))
    py = xp.asarray(np.random.uniform(y_min + 0.5, y_max - 0.5, num_particles))

    def to_pixel(phys_x, phys_y):
        sx = (phys_x - x_min) / (x_max - x_min) * width
        sy = (y_max - phys_y) / (y_max - y_min) * height
        return sx, sy

    trail_history = []
    max_trail_len = 120
    variable_logs = []

    print("Generating Gradient Descent frames...")

    def frame_generator():
        nonlocal px, py
        for f in range(num_frames):
            gx, gy = himmelblau_grad(px, py)
            mag = xp.sqrt(gx**2 + gy**2) + 1e-8
        
            # Log variables using Priority 1 roles
            mean_loss = float(xp.mean(himmelblau(px, py)))
            mean_grad = float(xp.mean(mag))
            variable_logs.append([
                {"name": "Avg Loss", "value": f"{mean_loss:.2f}", "role": "metric", "metric_index": 0},
                {"name": "Avg Gradient", "value": f"{mean_grad:.2f}", "role": "metric", "metric_index": 1},
                {"name": "Clip Bound", "value": "5.0", "role": "control"}
            ])
        
            scale = xp.minimum(mag, 5.0) / mag
            px = px - learning_rate * gx * scale
            py = py - learning_rate * gy * scale

            if hasattr(px, "get"):
                px_cpu, py_cpu = px.get(), py.get()
            else:
                px_cpu, py_cpu = np.asarray(px), np.asarray(py)

            sx, sy = to_pixel(px_cpu, py_cpu)

            trail_history.append((sx.copy(), sy.copy()))
            if len(trail_history) > max_trail_len:
                trail_history.pop(0)

            img = Image.fromarray(bg_cpu.copy(), "RGB").convert("RGBA")
            draw = ImageDraw.Draw(img)

            history_len = len(trail_history)
            if history_len > 1:
                for h in range(1, history_len):
                    opacity = int(100 * (h / history_len) ** 1.5)
                    prev_sx, prev_sy = trail_history[h - 1]
                    curr_sx, curr_sy = trail_history[h]
                    for i in range(num_particles):
                        col = particle_color(i, num_particles)
                        draw.line(
                            [(prev_sx[i], prev_sy[i]), (curr_sx[i], curr_sy[i])],
                            fill=col + (opacity,), width=2
                        )

            for i in range(num_particles):
                col = particle_color(i, num_particles)
                draw.ellipse([sx[i]-4, sy[i]-4, sx[i]+4, sy[i]+4], fill=col + (255,))

            yield np.array(img.convert("RGB"))

            if (f + 1) % 30 == 0:
                print(f"  Frame {f + 1}/{num_frames}")

    # Pedagogical annotations: Anchor to the first particle's starting position
    # and explicitly draw its gradient vector being clipped by the boundary.
    if hasattr(px, "get"):
        px0, py0 = float(px.get()[0]), float(py.get()[0])
    else:
        px0, py0 = float(px[0]), float(py[0])
    sx0, sy0 = to_pixel(px0, py0)
    
    gx, gy = himmelblau_grad(px0, py0)
    
    # Scale factor for visualization of the gradient space (pixels per unit of gradient)
    # The gradient magnitude starts around ~40-60.
    grad_vis_scale = 3.0
    
    # Unclipped gradient vector visualization
    # Pixel y-axis goes DOWN, while physical y-axis goes UP.
    # Gradient descent moves in -grad direction, so we visualize the negative gradient.
    end_x_unclipped = sx0 - gx * grad_vis_scale
    end_y_unclipped = sy0 + gy * grad_vis_scale 
    
    # Clipped gradient vector
    mag = np.sqrt(gx**2 + gy**2)
    scale = min(mag, 5.0) / mag
    end_x_clipped = sx0 - (gx * scale) * grad_vis_scale
    end_y_clipped = sy0 + (gy * scale) * grad_vis_scale
    
    clip_radius_px = 5.0 * grad_vis_scale
    
    annotations = [
        {
            "type": "circle",
            "coords": [sx0, sy0, clip_radius_px],
            "label": "Clip Bound (5.0)",
            "color": "control",
            "label_offset": (0, -18),  # pushed up so it clears the two line labels below
        },
        {
            "type": "line",
            "coords": [sx0, sy0, end_x_unclipped, end_y_unclipped],
            "label": "Raw ∇f",
            "color": "secondary",
            "label_offset": (35, 10),  # stacked below Clip Bound's label, pushed right to clear the circle outline
        },
        {
            "type": "line",
            "coords": [sx0, sy0, end_x_clipped, end_y_clipped],
            "label": "Clipped Step",
            "color": "primary",
            "label_offset": (0, 34),  # stacked below Raw ∇f's label
        }
    ]

    return frame_generator(), variable_logs, None, annotations
