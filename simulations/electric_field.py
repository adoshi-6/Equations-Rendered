import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import xp

def evaluate_point(inp):
    x, y = inp["x"], inp["y"]
    # charges: 1.0 at (-1.5, 0), -1.0 at (1.5, 0)
    q1, x1, y1 = 1.0, -1.5, 0.0
    q2, x2, y2 = -1.0, 1.5, 0.0
    
    k = 1.0
    dx1, dy1 = x - x1, y - y1
    dx2, dy2 = x - x2, y - y2
    r1 = max(np.sqrt(dx1**2 + dy1**2), 1e-6)
    r2 = max(np.sqrt(dx2**2 + dy2**2), 1e-6)
    
    Ex = k * q1 * dx1 / r1**3 + k * q2 * dx2 / r2**3
    Ey = k * q1 * dy1 / r1**3 + k * q2 * dy2 / r2**3
    return {"Ex": float(Ex), "Ey": float(Ey)}

TEST_SPEC = {
    "category": "known_points",
    "known_points": [
        ({"x": 0.0, "y": 0.0}, {"Ex": 2.0 / 1.5**2, "Ey": 0.0}, 1e-4), # midway
        ({"x": 1.0, "y": 1.0}, {"Ex": 0.4858366, "Ey": -0.6643154}, 1e-4),
        ({"x": -1.0, "y": -1.0}, {"Ex": 0.4858366, "Ey": -0.6643154}, 1e-4),
    ],
}

# Universal Palette
COLOR_ROLE_1 = (232, 93, 74)     # Primary (#E85D4A)
COLOR_ROLE_2 = (93, 168, 232)    # Secondary (#5DA8E8)
COLOR_ROLE_3 = (127, 174, 107)   # Auxiliary (#7FAE6B)
COLOR_TRAIL = (127, 174, 107)    # Auxiliary for fields

def recommended_duration(config: dict) -> float:
    """
    Computes duration for a full rotation of the charge configuration.
    10 seconds provides a smooth 360-degree rotation.
    """
    return 10.0

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Generates frames and variable logs for the Electric Field Lines simulation.
    """
    duration = config.get("duration", 10.0)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)

    width, height = 1080, 1080
    center_x, center_y = 540.0, 540.0

    charge_radius = 200.0
    charges_q = [1.0, -1.0, 1.0, -1.0]
    n_charges = len(charges_q)
    base_angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

    num_seeds = 160
    seed_angles = np.linspace(0, 2 * np.pi, num_seeds // 2, endpoint=False)

    trace_steps = 180
    trace_dt = 3.5
    variable_logs = []

    print("Generating Electric Field Lines frames...")

    def frame_generator():
        for f in range(num_frames):
            progress = f / max(1, num_frames - 1)
            rotation = progress * 2.0 * np.pi
        
            # Log variables
            variable_logs.append([
                {"name": "System Angle", "value": f"{(rotation * 180 / np.pi):.1f}°", "role": "metric", "metric_index": 0},
                {"name": "Total Charge", "value": "0.0 C", "role": "metric", "metric_index": 1}
            ])

            charge_positions = []
            for i in range(n_charges):
                angle = base_angles[i] + rotation
                cx = center_x + charge_radius * np.cos(angle)
                cy = center_y + charge_radius * np.sin(angle)
                charge_positions.append((cx, cy))

            seeds_x = []
            seeds_y = []
            for i, q in enumerate(charges_q):
                if q > 0:
                    cx, cy = charge_positions[i]
                    for sa in seed_angles:
                        seeds_x.append(cx + 18.0 * np.cos(sa))
                        seeds_y.append(cy + 18.0 * np.sin(sa))

            seeds_x = np.array(seeds_x)
            seeds_y = np.array(seeds_y)
            n_seeds = len(seeds_x)

            all_x = np.zeros((trace_steps, n_seeds))
            all_y = np.zeros((trace_steps, n_seeds))
            all_x[0] = seeds_x
            all_y[0] = seeds_y

            for step in range(1, trace_steps):
                px_s = all_x[step - 1]
                py_s = all_y[step - 1]

                ex = np.zeros(n_seeds)
                ey = np.zeros(n_seeds)

                for i, q in enumerate(charges_q):
                    cx, cy = charge_positions[i]
                    dx = px_s - cx
                    dy = py_s - cy
                    r2 = dx**2 + dy**2 + 1e-4
                    r = np.sqrt(r2)
                    r3 = r2 * r
                    ex += q * dx / r3
                    ey += q * dy / r3

                mag = np.sqrt(ex**2 + ey**2) + 1e-12
                ex /= mag
                ey /= mag

                all_x[step] = px_s + trace_dt * ex
                all_y[step] = py_s + trace_dt * ey

            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            draw = ImageDraw.Draw(img)

            for s in range(n_seeds):
                t = s / max(1, n_seeds - 1)
                if t < 0.5:
                    fac = t * 2.0
                    col = (
                        int(COLOR_ROLE_1[0] * (1-fac) + COLOR_TRAIL[0] * fac),
                        int(COLOR_ROLE_1[1] * (1-fac) + COLOR_TRAIL[1] * fac),
                        int(COLOR_ROLE_1[2] * (1-fac) + COLOR_TRAIL[2] * fac)
                    )
                else:
                    fac = (t - 0.5) * 2.0
                    col = (
                        int(COLOR_TRAIL[0] * (1-fac) + COLOR_ROLE_2[0] * fac),
                        int(COLOR_TRAIL[1] * (1-fac) + COLOR_ROLE_2[1] * fac),
                        int(COLOR_TRAIL[2] * (1-fac) + COLOR_ROLE_2[2] * fac)
                    )

                for step in range(1, trace_steps):
                    x0, y0 = all_x[step - 1, s], all_y[step - 1, s]
                    x1, y1 = all_x[step, s], all_y[step, s]

                    if (x1 < -50 or x1 > width + 50 or y1 < -50 or y1 > height + 50):
                        break

                    alpha = max(10, int(160 * (1.0 - step / trace_steps)))
                    draw.line([(x0, y0), (x1, y1)], fill=col + (alpha,), width=2)
                    
            for i, q in enumerate(charges_q):
                cx, cy = charge_positions[i]
                r_draw = 14
                if q > 0:
                    draw.ellipse([cx - r_draw, cy - r_draw, cx + r_draw, cy + r_draw],
                                 fill=COLOR_ROLE_1 + (255,), outline=(255, 200, 100, 255), width=2)
                    draw.line([(cx - 6, cy), (cx + 6, cy)], fill=(255, 255, 255, 255), width=2)
                    draw.line([(cx, cy - 6), (cx, cy + 6)], fill=(255, 255, 255, 255), width=2)
                else:
                    draw.ellipse([cx - r_draw, cy - r_draw, cx + r_draw, cy + r_draw],
                                 fill=COLOR_ROLE_2 + (255,), outline=(100, 180, 255, 255), width=2)
                    draw.line([(cx - 6, cy), (cx + 6, cy)], fill=(255, 255, 255, 255), width=2)

            # Pedagogical Annotation: Vector from a charge (r_i) to a fixed point in space (r)
            # representing \vec{r} - \vec{r}_i
            target_x, target_y = center_x + 150, center_y - 150
            cx0, cy0 = charge_positions[0] # Positive charge 0
            
            # Draw point \vec{r}
            draw.ellipse([target_x - 5, target_y - 5, target_x + 5, target_y + 5], fill=COLOR_ROLE_3 + (255,))
            
            # Draw vector \vec{r} - \vec{r}_i
            draw.line([(cx0, cy0), (target_x, target_y)], fill=COLOR_ROLE_3 + (255,), width=6)
            
            # Arrowhead
            dx_r, dy_r = target_x - cx0, target_y - cy0
            mag_r = np.sqrt(dx_r**2 + dy_r**2)
            if mag_r > 0:
                ux, uy = dx_r / mag_r, dy_r / mag_r
                draw.line([(target_x, target_y), (target_x - 15*ux - 15*uy, target_y - 15*uy + 15*ux)], fill=COLOR_ROLE_3 + (255,), width=5)
                draw.line([(target_x, target_y), (target_x - 15*ux + 15*uy, target_y - 15*uy - 15*ux)], fill=COLOR_ROLE_3 + (255,), width=5)
                
            # Label r
            # Use matplotlib to generate the label once
            if not hasattr(frame_generator, "math_label"):
                import matplotlib.pyplot as plt
                import io
                fig = plt.figure(figsize=(2, 1), dpi=100)
                fig.patch.set_alpha(0.0)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ax.patch.set_alpha(0.0)
                # Use COLOR_ROLE_3 hex for color
                hex_color = '#%02x%02x%02x' % COLOR_ROLE_3
                ax.text(0.5, 0.5, r"$\vec{r} - \vec{r}_i$", color=hex_color, fontsize=32, ha='center', va='center', weight='bold')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                frame_generator.math_label = Image.open(buf).convert("RGBA")
                
            label_x = int((cx0 + target_x) / 2 + 15)
            label_y = int((cy0 + target_y) / 2 - 25)
            
            # Paste the math label
            img.paste(frame_generator.math_label, (label_x, label_y), frame_generator.math_label)

            yield np.array(img.convert("RGB"))

            if (f + 1) % 30 == 0:
                print(f"  Frame {f + 1}/{num_frames}")

    return frame_generator(), variable_logs, None
