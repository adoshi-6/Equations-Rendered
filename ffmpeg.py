import subprocess
import os

def compile_video(frame_pattern: str, audio_path: str | None, output_path: str, fps: int = 30) -> None:
    """
    Compiles a sequence of PNG frames and an optional audio file into an MP4 video using FFmpeg.
    
    Parameters:
    - frame_pattern: The path pattern for the input frames (e.g., "temp_frames/frame_%04d.png").
    - audio_path: The path to the background audio track (MP3 or WAV), or None if silent.
    - output_path: The path where the final MP4 video will be written.
    - fps: Frame rate of the output video.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", frame_pattern]
    
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-stream_loop", "-1", "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    else:
        # If no audio or audio doesn't exist, just encode the video
        cmd.extend(["-map", "0:v:0"])
        
    # Standard settings for high compatibility (H.264, YUV420p)
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # High quality, low loss
        output_path
    ])
    
    print(f"Running ffmpeg command: {' '.join(cmd)}")
    
    # Run the process
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg compilation failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    else:
        print(f"Successfully compiled video to {output_path}")

def start_ffmpeg_process(output_path: str, fps: int = 30, audio_path: str | None = None, width: int = 1080, height: int = 1920):
    """
    Starts an FFmpeg process configured to read raw RGB frames from stdin.
    Returns (Popen object, log_file_path) so the caller can pipe frames via
    process.stdin.write(raw_bytes), and after the process exits, read the
    log file for diagnostics or close it explicitly.

    NOTE: previously this function opened `log_file` and passed it as
    `stderr=log_file` to Popen, but never returned or otherwise exposed the
    file handle to the caller — it was never explicitly closed. Additionally,
    `Popen.stderr` is None whenever stderr is redirected to a real file
    (rather than `subprocess.PIPE`), so renderer.py's failure-handling code
    (`ffmpeg_process.stderr.read() if ffmpeg_process.stderr else "Unknown
    error"`) always silently fell through to "Unknown error" on failure,
    even though the real ffmpeg error message was sitting in the log file on
    disk the whole time. Fixed by returning the log file path so the caller
    can read it back on failure and close the handle explicitly on success.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    cmd = [
        "ffmpeg", "-y", 
        "-f", "rawvideo", 
        "-vcodec", "rawvideo", 
        "-s", f"{width}x{height}", 
        "-pix_fmt", "rgb24", 
        "-framerate", str(fps), 
        "-i", "-"
    ]
    
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-stream_loop", "-1", "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
    else:
        cmd.extend(["-map", "0:v:0"])
        
    cmd.extend([
        "-loglevel", "error",  # Reduce output to prevent pipe blocking
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path
    ])
    
    log_file_path = output_path + ".ffmpeg.log"
    log_file = open(log_file_path, "w")
    print(f"Starting ffmpeg stream: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_file)
    # Stash the handle on the process object so the caller can close it
    # explicitly once the process has exited (can't close it here — ffmpeg
    # is still writing to it as frames stream in).
    process._claude_log_file = log_file
    process._claude_log_path = log_file_path
    return process
