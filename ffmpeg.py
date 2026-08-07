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
