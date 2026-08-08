import numpy as np

def generate(config: dict) -> tuple[list[np.ndarray], list[dict]]:
    """
    Trivial dummy simulation that generates a moving color gradient.
    Conforms to the contract: returns a list of RGB numpy arrays.
    """
    duration = config.get("duration", 3)
    fps = config.get("fps", 30)
    num_frames = int(duration * fps)
    
    variable_logs = []
    
    # Grid coordinates
    x = np.linspace(0, 1, 1080)
    y = np.linspace(0, 1, 1080)
    xx, yy = np.meshgrid(x, y)
    
    def frame_generator():
        for i in range(num_frames):
            frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
        
            # Color coefficients that evolve over time
            t_factor = i / num_frames
        
            # Generate vectorized channels
            # R: horizontal sweep
            # G: vertical sweep
            # B: temporal oscillation
            r = (xx * 255 * t_factor).astype(np.uint8)
            g = (yy * 255 * (1.0 - t_factor)).astype(np.uint8)
            b = (np.ones((1080, 1080)) * (128 + 127 * np.sin(i * 0.1))).astype(np.uint8)
        
            frame[:, :, 0] = r
            frame[:, :, 1] = g
            frame[:, :, 2] = b
        
            yield frame
        
    return frame_generator(), variable_logs, None
