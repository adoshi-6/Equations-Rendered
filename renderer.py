import os
import sys
import yaml
import argparse
import importlib
import shutil
import json
import datetime
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont

# Same UTF-8 stdout fix as tests/run_physics_tests.py — several simulations
# print names/status text containing non-ASCII characters (e.g. "Rössler
# Attractor"), which crashes on Windows' default cp1252 console encoding
# rather than only ever running under a UTF-8-default environment.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
        raise RuntimeError(f"Equation rendering failed! Matplotlib is required but encountered an error: {e}")

def render_video(config_path: str, output_path: str, baseline_dir: str = None) -> None:
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
    
    # Apply tuning adjustment: +5s buffer AFTER plateau & clamp for chaotic ODEs
    if sim_name in ["double_pendulum", "lorenz", "rossler", "three_body"]:
        duration += 5.0
        duration = min(30.0, duration)  # Ensure we still respect the 30s ceiling
        
    config["duration"] = duration  # Update config so generate() uses the clamped duration
    fps = config.get("fps", 30)
        
    print(f"Running simulation: {sim_name} (Duration: {duration}s)...")
    sim_output = sim_module.generate(config)
    
    if isinstance(sim_output, tuple):
        if len(sim_output) == 3:
            frame_gen, variable_logs, auxiliary_curves = sim_output
        elif len(sim_output) == 4:
            frame_gen, variable_logs, auxiliary_curves, _ = sim_output
        else:
            raise ValueError("Simulation generate() must return 3 or 4 elements.")
    else:
        raise ValueError("Simulation generate() must return (frame_generator, variable_logs, auxiliary_curves_or_None, [annotations]).")
        
    num_frames = int(duration * fps)
    print(f"Simulation will generate ~{num_frames} frames.")
    
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

    # 3. Find audio track (if any) in assets/music before streaming
    music_dir = "assets/music"
    audio_path = None
    if os.path.exists(music_dir):
        music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) 
                       if f.lower().endswith(('.mp3', '.wav'))]
        if music_files:
            audio_path = music_files[0]
            print(f"Found background audio: {audio_path}")

    # 4. Start FFmpeg process for streaming
    ffmpeg_process = ffmpeg.start_ffmpeg_process(output_path, fps=fps, audio_path=audio_path)
    
    # 5. Composite frames dynamically
    title_font = load_font(84)
    readout_font = load_font(42)
    curve_font = load_italic_font(32)
    annotation_font = load_italic_font(28)
    
    ROLE_COLORS = {
        "primary": "#E85D4A",      # soft red — main trajectory/body
        "secondary": "#5DA8E8",    # powder blue — comparison trajectory
        "auxiliary": "#7FAE6B",    # sage green — derived/aux curve
        "control": "#D4C24A",      # muted yellow — tunable parameter
        "static": "#A8B5C2",       # pale blue-grey — fixed parameter
    }
    
    METRIC_COLORS = [
        "#E8905D",  # warm amber — metric 0
        "#B87FC9",  # dusty purple — metric 1
        "#C97FA0",  # muted rose — metric 2 (if needed)
        "#7FC9B0",  # muted teal — metric 3 (if needed)
    ]
    
    # Check if annotations are returned
    annotations = None
    if isinstance(sim_output, tuple) and len(sim_output) == 4:
        annotations = sim_output[3]
    
    # Pre-calculate bounding boxes for provenance
    provenance_bboxes = {}
    
    print("Streaming and compositing frames...")
    
    start_idx = max(0, int(num_frames * 0.1))
    mid_idx = max(0, int(num_frames * 0.5))
    end_idx = max(0, num_frames - 1)
    
    last_canvas = None
    
    for i, frame in enumerate(frame_gen):
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
            # Revert to original size (1080x1080)
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
        
        # Store title bbox on first frame
        if i == 0:
            provenance_bboxes["title_bbox"] = {"y1": ty, "y2": ty + 90, "x1": tx, "x2": tx + title_w} # rough height estimate
        
        # Draw Variable Log Readout (below title)
        try:
            if variable_logs:
                current_log = variable_logs[-1]
                if current_log:
                    if isinstance(current_log, dict):
                        # Legacy fallback for simulations not yet updated to list-of-dicts
                        log_str = " | ".join(f"{k}: {v}" for k, v in current_log.items())
                        log_w = draw.textlength(log_str, font=readout_font)
                        draw.text(((1080 - log_w) // 2, 280), log_str, font=readout_font, fill=(0, 255, 0))
                    elif isinstance(current_log, list):
                        # New Priority 1: Color-coded variables
                        # First pass: calculate total width
                        total_w = 0
                        segments = []
                        for idx, entry in enumerate(current_log):
                            name = entry.get("name", "")
                            value = entry.get("value", "")
                            role = entry.get("role", "static")
                            if role == "metric":
                                metric_index = entry.get("metric_index", 0)
                                color = METRIC_COLORS[metric_index % len(METRIC_COLORS)]
                            else:
                                color = ROLE_COLORS.get(role, "#CFCFCF")
                            
                            text_str = f"{name}: {value}"
                            w = draw.textlength(text_str, font=readout_font)
                            segments.append({"text": text_str, "width": w, "color": color})
                            total_w += w
                            
                            if idx < len(current_log) - 1:
                                pipe_str = "  |  "
                                pipe_w = draw.textlength(pipe_str, font=readout_font)
                                segments.append({"text": pipe_str, "width": pipe_w, "color": "#FFFFFF"})
                                total_w += pipe_w
                            
                        # Second pass: draw segments centered at fixed y=280
                        curr_x = (1080 - total_w) // 2
                        for seg in segments:
                            draw.text((curr_x, 280), seg["text"], font=readout_font, fill=seg["color"])
                            curr_x += seg["width"]
        except Exception as e:
            print(f"Exception during readout rendering: {e}")
            import traceback
            traceback.print_exc()

        # Draw Annotations
        if annotations:
            try:
                for ann in annotations:
                    color_hex = ROLE_COLORS.get(ann.get("color", "static"), "#CFCFCF")
                    coords = ann.get("coords", [])
                    label = ann.get("label", "")
                    ann_type = ann.get("type", "line")
                    
                    if ann_type == "circle" and len(coords) == 3:
                        cx, cy, r = coords
                        # Adjust coordinates relative to the simulation bounding box
                        # since sim_img is pasted at (sim_x, sim_y)
                        canvas_cx = sim_x + cx
                        canvas_cy = sim_y + cy
                        draw.ellipse([canvas_cx - r, canvas_cy - r, canvas_cx + r, canvas_cy + r], 
                                     outline=color_hex, width=2)
                        
                        # Draw a small line to the label
                        label_dx, label_dy = ann.get("label_offset", (0, 0))
                        draw.line([canvas_cx + r, canvas_cy, canvas_cx + r + 20, canvas_cy], fill=color_hex, width=2)
                        draw.text((canvas_cx + r + 25 + label_dx, canvas_cy - 15 + label_dy), label, font=annotation_font, fill=color_hex)
                        
                    elif ann_type == "bracket" and len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        x1, x2 = sim_x + x1, sim_x + x2
                        y1, y2 = sim_y + y1, sim_y + y2
                        draw.line([x1, y1, x2, y2], fill=color_hex, width=2)
                        # Add bracket ticks
                        draw.line([x1, y1-5, x1, y1+5], fill=color_hex, width=2)
                        draw.line([x2, y2-5, x2, y2+5], fill=color_hex, width=2)
                        # Label at midpoint
                        label_dx, label_dy = ann.get("label_offset", (0, 0))
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        draw.text((mx + 10 + label_dx, my - 15 + label_dy), label, font=annotation_font, fill=color_hex)

                    elif ann_type == "line" and len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        cx1, cy1 = sim_x + x1, sim_y + y1
                        cx2, cy2 = sim_x + x2, sim_y + y2
                        draw.line([cx1, cy1, cx2, cy2], fill=color_hex, width=3)

                        # Arrowhead at the end point (cx2, cy2)
                        dx_l, dy_l = cx2 - cx1, cy2 - cy1
                        mag_l = (dx_l ** 2 + dy_l ** 2) ** 0.5
                        if mag_l > 1e-6:
                            ux_l, uy_l = dx_l / mag_l, dy_l / mag_l
                            ahx1 = cx2 - 12 * ux_l - 8 * uy_l
                            ahy1 = cy2 - 12 * uy_l + 8 * ux_l
                            ahx2 = cx2 - 12 * ux_l + 8 * uy_l
                            ahy2 = cy2 - 12 * uy_l - 8 * ux_l
                            draw.line([cx2, cy2, ahx1, ahy1], fill=color_hex, width=3)
                            draw.line([cx2, cy2, ahx2, ahy2], fill=color_hex, width=3)

                        # Label near the midpoint, offset perpendicular to the line.
                        # `label_offset` (dx, dy) lets a simulation stagger labels
                        # apart when multiple annotations share a small anchor
                        # region — added after gradient_descent's Raw ∇f/Clipped
                        # Step/Clip Bound labels were found to overlap illegibly
                        # once the "line" annotation type actually started
                        # rendering (previously silently dropped, so this
                        # collision was never visible before).
                        label_dx, label_dy = ann.get("label_offset", (0, 0))
                        mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                        draw.text((mx + 10 + label_dx, my - 15 + label_dy), label, font=annotation_font, fill=color_hex)

            except Exception as e:
                print(f"Exception during annotation rendering: {e}")
        
        # Draw Equation Overlay
        max_eq_width = 1000
        eq_scaled = eq_overlay.copy()
        eq_w, eq_h = eq_scaled.size
        if eq_w > max_eq_width:
            scale_factor = max_eq_width / eq_w
            new_w = int(eq_w * scale_factor)
            new_h = int(eq_h * scale_factor)
            eq_scaled = eq_scaled.resize((new_w, new_h), Image.Resampling.LANCZOS)
            eq_w, eq_h = eq_scaled.size
        
        eq_x = (1080 - eq_w) // 2
        # Revert equation y position
        target_eq_y = 1780 if auxiliary_curves is not None else 1600
        eq_y = target_eq_y - eq_h // 2
        
        if eq_y + eq_h > 1920 - 40:
            eq_y = 1920 - 40 - eq_h
        
        canvas.paste(eq_scaled, (eq_x, eq_y), eq_scaled)
        
        if i == 0:
            provenance_bboxes["equation_bbox"] = {"y1": eq_y, "y2": eq_y + eq_h, "x1": eq_x, "x2": eq_x + eq_w}
        
        # Stream bytes directly to ffmpeg
        try:
            ffmpeg_process.stdin.write(canvas.tobytes())
        except BrokenPipeError:
            print("Error: FFmpeg pipe broken! Stopping frame stream.")
            break
            
        last_canvas = canvas
        
        # Save baseline frames for test_visuals.py
        if baseline_dir and i in (start_idx, mid_idx, end_idx):
            os.makedirs(baseline_dir, exist_ok=True)
            name = "start" if i == start_idx else "mid" if i == mid_idx else "end"
            dst = os.path.join(baseline_dir, f"{sim_name}_{name}.png")
            canvas.save(dst)
            
        # Save explicit percentage frames for the user review (10%, 50%, 90%)
        # NOTE: computed dynamically from the ACTUAL num_frames of this render,
        # not hardcoded — a fixed frame-index dict (e.g. {27: "10pct", ...})
        # silently drifts out of sync whenever duration/num_frames changes
        # between renders (confirmed bug: fixed indices assumed ~276 frames,
        # but renders now commonly produce 300+ frames after the 10s floor
        # clamp, making "frame 27" actually ~9% instead of 10%, etc.)
        if baseline_dir:
            pct_checkpoints = {
                max(0, int(num_frames * 0.10)): "10pct",
                max(0, int(num_frames * 0.50)): "50pct",
                max(0, num_frames - 1 - int(num_frames * 0.10)): "90pct",
            }
            if i in pct_checkpoints:
                name = pct_checkpoints[i]
                dst = os.path.join(baseline_dir, f"{sim_name}_frame_{i:03d}_{name}_v3.png")
                canvas.save(dst)
                print(f"Saved checkpoint frame {i} ({name}, of {num_frames} total) to {dst}")
            
    # Close the stdin pipe to tell ffmpeg we are done
    if ffmpeg_process.stdin:
        ffmpeg_process.stdin.close()
        
    print("Waiting for FFmpeg to finish...")
    ffmpeg_process.wait()

    # Close the stderr log file handle now that ffmpeg has exited (can't
    # close it earlier — ffmpeg writes to it throughout the run). Previously
    # this handle was opened in ffmpeg.py and never closed anywhere.
    log_file = getattr(ffmpeg_process, "_claude_log_file", None)
    log_path = getattr(ffmpeg_process, "_claude_log_path", None)
    if log_file:
        log_file.close()
    
    if ffmpeg_process.returncode != 0:
        # Read the actual error from the log file on disk — ffmpeg_process.stderr
        # is always None here since stderr was redirected to a file, not a
        # pipe, so the old `ffmpeg_process.stderr.read() if ... else "Unknown
        # error"` fallback always silently produced "Unknown error" even
        # though the real message was on disk the whole time.
        err_out = "Unknown error"
        if log_path and os.path.exists(log_path):
            with open(log_path, "r") as f:
                err_out = f.read() or "(ffmpeg log file was empty)"
        raise RuntimeError(f"FFmpeg failed with code {ffmpeg_process.returncode}\n{err_out}")
        
    print(f"Successfully compiled video to {output_path}")
    
    if last_canvas:
        last_canvas.save("output/last_frame.png")
        print("Saved last frame to output/last_frame.png")
                
    # Generate provenance sidecar
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown"
        
    provenance = {
        "simulation": sim_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "commit": commit_hash,
        "environment": {
            "python": sys.version,
            "os": os.name
        },
        "ocr_bboxes": provenance_bboxes
    }
    prov_path = os.path.join(os.path.dirname(output_path), f"{sim_name}_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=4)
        
    print(f"Generated provenance sidecar: {prov_path}")

        
    # 6. Cleanup temp files
    # Retry with backoff instead of a single attempt: on Windows, a
    # just-written folder can be transiently locked by a background process
    # (OneDrive sync indexing new files, antivirus scanning, etc.) for a
    # second or two after the render finishes writing to it — not a real
    # problem, just a timing race. A single rmtree attempt fails outright in
    # that window (confirmed: WinError 5 "Access is denied" when the repo
    # sits inside an actively-syncing OneDrive folder). Retrying a few times
    # with a short delay resolves this the vast majority of the time without
    # needing to know or care what's holding the lock.
    cleanup_succeeded = False
    last_error = None
    for attempt in range(5):
        try:
            shutil.rmtree(temp_dir)
            cleanup_succeeded = True
            break
        except Exception as e:
            last_error = e
            if attempt < 4:
                time.sleep(0.5 * (attempt + 1))  # 0.5s, 1.0s, 1.5s, 2.0s

    if cleanup_succeeded:
        print("Cleaned up temporary rendering directory.")
    else:
        print(f"Warning: Could not clean up temporary folder {temp_dir} after 5 attempts: {last_error}")
        print("This does not affect the render output — only leftover temp files remain.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render simulation equations to video.")
    parser.path = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output", required=True, help="Path to save the output MP4 video.")
    parser.add_argument("--baseline-dir", required=False, help="Directory to save Start/Mid/End baseline frames.")
    
    args = parser.parse_args()
    render_video(args.config, args.output, args.baseline_dir)
