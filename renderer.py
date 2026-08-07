import os
import sys
import yaml
import argparse
import importlib
import shutil
from PIL import Image, ImageDraw, ImageFont

# Add current directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ffmpeg

def load_italic_font(font_size: int):
    font_paths = [
        "C:\\Windows\\Fonts\\timesi.ttf",  # Times New Roman Italic
        "C:\\Windows\\Fonts\\georgiai.ttf", # Georgia Italic
        "C:\\Windows\\Fonts\\cambriaz.ttf", # Cambria Italic
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return load_font(font_size)

def load_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """
    Attempts to load a premium font from standard paths, falling back to default PIL font.
    """
    font_paths = [
        "C:\\Windows\\Fonts\\timesbd.ttf",  # Times New Roman Bold
        "C:\\Windows\\Fonts\\times.ttf",    # Times New Roman
        "C:\\Windows\\Fonts\\segoeui.ttf",  # Segoe UI (Windows)
        "C:\\Windows\\Fonts\\arial.ttf",    # Arial (Windows)
        "assets/fonts/Outfit-Regular.ttf",
        "assets/fonts/Inter-Regular.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception as e:
                print(f"Font loading failed for {path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    print("Falling back to default font!")
    return ImageFont.load_default()

def render_equation_latex(equation: str, output_path: str) -> bool:
    """
    Renders an equation to a transparent PNG.
    Fallback chain:
    1. Manim's MathTex (primary choice)
    2. Matplotlib's mathtext (self-contained LaTeX engine)
    3. PIL draw text (plain text fallback)
    """
    print(f"Rendering equation: {equation}")
    
    # Attempt Matplotlib rendering (self-contained math engine)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Format the TeX formula correctly for Matplotlib
        eq_str = equation.strip()
        if not eq_str.startswith('$'):
            eq_str = fr"${eq_str}$"
            
        fig = plt.figure(figsize=(8, 2.5), dpi=200)
        fig.patch.set_alpha(0.0)
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        
        # Draw the text in white
        ax.text(0.5, 0.5, eq_str, color='white', fontsize=28, 
                horizontalalignment='center', verticalalignment='center')
        
        plt.savefig(output_path, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print("Successfully rendered equation via Matplotlib.")
        return True
        
    except Exception as e:
        import traceback
        print(f"Matplotlib rendering failed: {e}")
        traceback.print_exc()
        # Fallback to PIL is only used if Matplotlib genuinely fails, 
        # but user said skip straight to mathtext, ONLY fallback if it genuinely fails.

    # 3. PIL Fallback (draw plain text on transparent image)
    try:
        # Create a transparent PIL image
        img = Image.new("RGBA", (1080, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = load_font(32)
        
        # Draw the equation centered
        text_w = draw.textlength(equation, font=font)
        draw.text(((1080 - text_w) // 2, 80), equation, font=font, fill=(255, 255, 255, 255))
        img.save(output_path)
        print("Successfully rendered equation via PIL.")
        return True
    except Exception as e3:
        print(f"PIL fallback rendering failed: {e3}")
        return False

def render_video(config_path: str, output_path: str) -> None:
    """
    Orchestrates the entire rendering pipeline:
    1. Loads configuration.
    2. Runs the simulation to generate frames.
    3. Composites each frame onto a 1080x1920 black canvas with title and equation.
    4. Compiles using FFmpeg.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    title = config.get("title", "Simulation")
    equation = config.get("equation", "")
    sim_name = config.get("simulation", "")
    # 1. Resolve and load the simulation module
    try:
        sim_module = importlib.import_module(f"simulations.{sim_name}")
    except ImportError as e:
        raise ImportError(f"Could not import simulation module '{sim_name}': {e}")
        
    # Auto-computed duration clamping [10s, 30s]
    if hasattr(sim_module, "recommended_duration"):
        base_duration = sim_module.recommended_duration(config)
    else:
        base_duration = 10.0
        
    if "duration" in config:
        duration = float(config["duration"])
    else:
        duration = base_duration
        
    duration = max(10.0, min(30.0, duration))
    config["duration"] = duration  # Update config so generate() uses the clamped duration
    fps = config.get("fps", 30)
        
    print(f"Running simulation: {sim_name} (Duration: {duration}s)...")
    sim_output = sim_module.generate(config)
    
    auxiliary_curves = None
    if isinstance(sim_output, tuple):
        if len(sim_output) == 3:
            sim_frames, variable_logs, auxiliary_curves = sim_output
        elif len(sim_output) == 2:
            sim_frames, variable_logs = sim_output
            
        if not sim_frames:
            raise ValueError("Simulation returned an empty list of frames.")
    else:
        sim_frames = sim_output
        variable_logs = [{}] * len(sim_frames)
        
    print(f"Simulation generated {len(sim_frames)} frames.")
    
    if not sim_frames:
        raise ValueError("Simulation returned an empty list of frames.")
        
    # 2. Render the equation overlay PNG (unique temp dir per simulation to avoid collisions)
    temp_dir = os.path.join(os.path.dirname(output_path), f"temp_render_{sim_name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    eq_overlay_path = os.path.join(temp_dir, "equation.png")
    render_equation_latex(equation, eq_overlay_path)
    
    eq_overlay = Image.open(eq_overlay_path).convert("RGBA")
    
    # 2.5 Generate static auxiliary plot background
    aux_bg_img = None
    lines_pixels = {}
    if auxiliary_curves is not None:
        import matplotlib.pyplot as plt
        import io
        import numpy as np
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=100) # 1080x500
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')
        
        time_arr = auxiliary_curves["time"]
        for series_name, series_info in auxiliary_curves["series"].items():
            ax.plot(time_arr, series_info["data"], alpha=0) # invisible just for bounds
            
        ax.axis('off')
        
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        ax.annotate('', xy=(x_max, 0), xytext=(x_min, 0), arrowprops=dict(arrowstyle="->", color='#888888', lw=1.5))
        ax.annotate('', xy=(0, y_max), xytext=(0, y_min), arrowprops=dict(arrowstyle="->", color='#888888', lw=1.5))
        
        ax.text(x_max, -0.05 * (y_max - y_min), auxiliary_curves.get("xlabel", "Time"), color='#888888', ha='right', va='top', fontstyle='italic', family='serif', size=12)
        ax.text(0.02 * (x_max - x_min), y_max, auxiliary_curves.get("ylabel", "Amplitude"), color='#888888', ha='left', va='top', fontstyle='italic', family='serif', size=12)
        
        plt.tight_layout(pad=2.0)
        
        # Get pixel coordinates for all points
        for series_name, series_info in auxiliary_curves["series"].items():
            points = np.column_stack((time_arr, series_info["data"]))
            pixels = ax.transData.transform(points)
            # Matplotlib origin is bottom-left; PIL origin is top-left
            pixels[:, 1] = 500 - pixels[:, 1]
            lines_pixels[series_name] = [(px, py) for px, py in pixels]
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        aux_bg_img = Image.open(buf).convert("RGBA")
        plt.close(fig)

    # 3. Composite frames
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    title_font = load_font(84)
    readout_font = load_font(42)
    curve_font = load_italic_font(32)
    
    print("Compositing frames...")
    for i, frame in enumerate(sim_frames):
        # Create base canvas (1080x1920 black image)
        canvas = Image.new("RGB", (1080, 1920), "black")
        
        # Convert simulation numpy array to PIL Image
        sim_img = Image.fromarray(frame)
        
        # If auxiliary curves are present, shrink main simulation to 800x800 and place higher
        if auxiliary_curves is not None:
            sim_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            sim_w, sim_h = sim_img.size
            sim_x = (1080 - sim_w) // 2
            sim_y = 350 # Place higher to make room for plot
            
            # Draw auxiliary plot
            frame_aux = aux_bg_img.copy()
            draw_aux = ImageDraw.Draw(frame_aux)
            
            for series_name, series_info in auxiliary_curves["series"].items():
                pts = lines_pixels[series_name][:i+1]
                if len(pts) > 1:
                    color = series_info.get("color", "white")
                    draw_aux.line(pts, fill=color, width=3, joint="curve")
                    end_px, end_py = pts[-1]
                    label_x = end_px + 10
                    # Prevent overlap with the right-aligned 'Time' axis label
                    label_y = end_py - 15
                    if label_x > 900:
                        label_x = end_px - 150
                    if label_x > 800 and label_y > 400:
                        label_y -= 40
                    draw_aux.text((label_x, label_y), series_name, fill=color, font=curve_font)
                    
            canvas.paste(frame_aux, (0, 1150), frame_aux)
        else:
            # Resize to fit within 1080x1080 box, maintaining aspect ratio
            sim_img.thumbnail((1080, 1080), Image.Resampling.LANCZOS)
            sim_w, sim_h = sim_img.size
            sim_x = (1080 - sim_w) // 2
            sim_y = (1920 - sim_h) // 2
            
        canvas.paste(sim_img, (sim_x, sim_y))
        
        # Draw Title
        draw = ImageDraw.Draw(canvas)
        title_w = draw.textlength(title, font=title_font)
        # Position title centered at y = 160
        tx = (1080 - title_w) // 2
        ty = 160
        draw.text((tx, ty), title, font=title_font, fill=(255, 255, 255))
        
        # Draw Variable Log Readout (Green text below title)
        try:
            if i < len(variable_logs) and variable_logs[i]:
                log_str = " | ".join(f"{k}: {v}" for k, v in variable_logs[i].items())
                log_w = draw.textlength(log_str, font=readout_font)
                draw.text(((1080 - log_w) // 2, 280), log_str, font=readout_font, fill=(0, 255, 0))
        except Exception as e:
            print(f"Exception during readout rendering: {e}")
            import traceback
            traceback.print_exc()
        
        # Draw Equation Overlay
        # Scale down if the equation image is wider than the canvas (with padding)
        max_eq_width = 1000
        eq_scaled = eq_overlay.copy()
        eq_w, eq_h = eq_scaled.size
        if eq_w > max_eq_width:
            scale_factor = max_eq_width / eq_w
            new_w = int(eq_w * scale_factor)
            new_h = int(eq_h * scale_factor)
            eq_scaled = eq_scaled.resize((new_w, new_h), Image.Resampling.LANCZOS)
            eq_w, eq_h = eq_scaled.size
        
        # Center equation horizontally
        eq_x = (1080 - eq_w) // 2
        
        # Target y = 1600 normally, or 1780 if we have aux plot taking up space
        target_eq_y = 1780 if auxiliary_curves is not None else 1600
        eq_y = target_eq_y - eq_h // 2
        
        # Clamp so equation never extends past the bottom (40px padding)
        if eq_y + eq_h > 1920 - 40:
            eq_y = 1920 - 40 - eq_h
        
        canvas.paste(eq_scaled, (eq_x, eq_y), eq_scaled)
        
        # Save composite frame
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        canvas.save(frame_path)
        
    print("Composition complete.")
    
    # 4. Find audio track (if any) in assets/music
    music_dir = "assets/music"
    audio_path = None
    if os.path.exists(music_dir):
        music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) 
                       if f.lower().endswith(('.mp3', '.wav'))]
        if music_files:
            audio_path = music_files[0]
            print(f"Found background audio: {audio_path}")
            
    # 5. Compile final video
    frame_pattern = os.path.join(frames_dir, "frame_%04d.png")
    ffmpeg.compile_video(frame_pattern, audio_path, output_path, fps=fps)
    
    # Save the last frame before cleanup
    last_frame_src = os.path.join(frames_dir, f"frame_{len(sim_frames)-1:04d}.png")
    if os.path.exists(last_frame_src):
        shutil.copy(last_frame_src, "output/last_frame.png")
        print("Saved last frame to output/last_frame.png")
        
    # 6. Cleanup temp files
    try:
        shutil.rmtree(temp_dir)
        print("Cleaned up temporary rendering directory.")
    except Exception as e:
        print(f"Warning: Could not clean up temporary folder {temp_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render simulation equations to video.")
    parser.path = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output", required=True, help="Path to save the output MP4 video.")
    
    args = parser.parse_args()
    render_video(args.config, args.output)
