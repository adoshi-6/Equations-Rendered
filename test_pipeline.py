import os
import wave
import struct
import sys

# Change working directory to the project directory to keep relative paths consistent
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.append(project_dir)

# Create silent WAV
music_dir = "assets/music"
os.makedirs(music_dir, exist_ok=True)
wav_path = os.path.join(music_dir, "silent_placeholder.wav")

if not os.path.exists(wav_path):
    print("Generating silent placeholder WAV...")
    duration = 3.0
    sample_rate = 44100
    num_frames = int(duration * sample_rate)
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setparams((1, 2, sample_rate, num_frames, 'NONE', 'not compressed'))
        silence = struct.pack('<h', 0) * num_frames
        wav_file.writeframes(silence)
    print(f"Generated WAV at {wav_path}")

# Run rendering
from renderer import render_video

config_path = "configs/dummy.yaml"
output_path = "output/dummy.mp4"

print("Starting video rendering...")
render_video(config_path, output_path)
print("Finished video rendering!")
