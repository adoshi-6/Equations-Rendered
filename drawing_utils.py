import math
from PIL import ImageDraw, ImageFont

def draw_hatching(draw: ImageDraw.Draw, p1: tuple, p2: tuple, normal_dir: tuple, spacing: int = 15, length: int = 15, color: tuple = (255, 255, 255), width: int = 2):
    """
    Draws a fixed wall line between p1 and p2, with diagonal hatch marks pointing away from normal_dir.
    """
    # Main wall line
    draw.line([p1, p2], fill=color, width=width)
    
    # Calculate direction vector of the line
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_len = math.hypot(dx, dy)
    
    if line_len < 1e-5:
        return
        
    ux = dx / line_len
    uy = dy / line_len
    
    # Calculate hatching direction (diagonal to normal, usually roughly 45 degrees)
    # normal_dir should be a unit vector pointing to the "inside" of the wall.
    # We want hatch lines pointing "outside" (opposite to normal).
    nx, ny = normal_dir
    
    # Normalize normal
    n_len = math.hypot(nx, ny)
    if n_len > 1e-5:
        nx /= n_len
        ny /= n_len
    
    # Hatching direction vector (45 deg angle combining -normal and tangent)
    hx = -nx * 0.707 + ux * 0.707
    hy = -ny * 0.707 + uy * 0.707
    
    num_hatches = int(line_len / spacing)
    for i in range(num_hatches + 1):
        t = i * spacing / line_len if line_len > 0 else 0
        t = min(t, 1.0)
        px = p1[0] + t * dx
        py = p1[1] + t * dy
        
        hx_end = px + hx * length
        hy_end = py + hy * length
        
        draw.line([(px, py), (hx_end, hy_end)], fill=color, width=1)

def draw_spring(draw: ImageDraw.Draw, p1: tuple, p2: tuple, num_coils: int = 10, width_spring: float = 20.0, color: tuple = (255, 255, 255), width: int = 2):
    """
    Draws a zigzag spring between p1 and p2.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    L = math.hypot(dx, dy)
    
    if L < 1e-5:
        draw.line([p1, p2], fill=color, width=width)
        return
        
    ux = dx / L
    uy = dy / L
    
    nx = -uy
    ny = ux
    
    # Spring consists of: flat start (10%), coils (80%), flat end (10%)
    points = []
    
    flat_len = max(5, L * 0.1)
    coil_len = max(0, L - 2 * flat_len)
    
    p_start = (p1[0] + ux * flat_len, p1[1] + uy * flat_len)
    p_end = (p2[0] - ux * flat_len, p2[1] - uy * flat_len)
    
    points.append(p1)
    points.append(p_start)
    
    # Add zigzag points
    num_segments = num_coils * 2
    for i in range(num_segments + 1):
        t = i / num_segments if num_segments > 0 else 0
        cx = p_start[0] + (p_end[0] - p_start[0]) * t
        cy = p_start[1] + (p_end[1] - p_start[1]) * t
        
        if i == 0 or i == num_segments:
            offset = 0
        else:
            offset = width_spring if i % 2 == 1 else -width_spring
            
        px = cx + nx * offset
        py = cy + ny * offset
        points.append((px, py))
        
    points.append(p_end)
    points.append(p2)
    
    draw.line(points, fill=color, width=width, joint="curve")

def draw_mass(draw: ImageDraw.Draw, center: tuple, size: tuple, label: str = "", font: ImageFont.FreeTypeFont = None, color: tuple = (255, 255, 255), outline: tuple = None, text_color: tuple = (0, 0, 0), width: int = 2):
    """
    Draws a rectangular mass box centered at 'center' with dimensions 'size' (width, height).
    """
    cx, cy = center
    w, h = size
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    if outline:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=outline, width=width)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=color)
        
    if label and font:
        # Measure text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # approximate center
        tx = cx - text_w / 2
        ty = cy - text_h / 2 - bbox[1]
        
        draw.text((tx, ty), label, fill=text_color, font=font)
