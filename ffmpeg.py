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

def start_ffmpeg_process(output_path: str, fps: int = 30, audio_path: str | None = None, width: int = 1080, height: int = 1920) -> subprocess.Popen:
    """
    Starts an FFmpeg process configured to read raw RGB frames from stdin.
    Returns the Popen object so frames can be piped into process.stdin.write(raw_bytes).
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
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_file)
