import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend import xp

class BoundingBoxPlateauDetector:
    def __init__(self, patience_steps: int = 50, dt: float = 0.05, rel_tolerance: float = 1e-3):
        """
        patience_steps: number of consecutive steps the cumulative extent must remain stable.
        dt: simulation timestep (used for logging actual simulated seconds).
        rel_tolerance: relative change threshold.
        """
        self.patience_steps = patience_steps
        self.dt = dt
        self.rel_tolerance = rel_tolerance
        self.history = []
        self.plateau_count = 0
        self.cum_min = None
        self.cum_max = None
        
    def check(self, coords):
        """
        coords: xp array of shape (N, D) where N is number of points/trajectories, D is dimensions.
        Returns True if the cumulative bounding box has plateaued.
        """
        min_vals = xp.min(coords, axis=0)
        max_vals = xp.max(coords, axis=0)
        
        if self.cum_min is None:
            self.cum_min = min_vals
            self.cum_max = max_vals
        else:
            self.cum_min = xp.minimum(self.cum_min, min_vals)
            self.cum_max = xp.maximum(self.cum_max, max_vals)
            
        extent = float(xp.sum(self.cum_max - self.cum_min))
        
        if not self.history:
            self.history.append(extent)
            return False
            
        last_extent = self.history[-1]
        
        # Avoid division by zero
        if last_extent > 1e-6:
            rel_change = abs(extent - last_extent) / last_extent
        else:
            rel_change = abs(extent - last_extent)
            
        if rel_change < self.rel_tolerance:
            self.plateau_count += 1
        else:
            self.plateau_count = 0
            
        self.history.append(extent)
        
        if self.plateau_count >= self.patience_steps:
            plateau_time = self.patience_steps * self.dt
            print(f"Plateau detected! Cumulative extent {extent:.3f} remained within {self.rel_tolerance*100}% for {self.patience_steps} steps ({plateau_time:.2f}s).")
            return True
            
        return False
