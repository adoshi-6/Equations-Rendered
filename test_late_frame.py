import sys
sys.path.append('.')
from simulations.double_pendulum import generate
import os
from PIL import Image

config = {"fps": 30, "duration": 28.0}
frames, logs = generate(config)

img = Image.fromarray(frames[-1])
img.save("dp_late_frame.png")
