import numpy as np


class PlateauTracker:
    def __init__(self, patience: int = 200, min_delta: float = 0.01, window: int = 100):
        self.patience = patience
        self.min_delta = min_delta
        self.window = window
        self.best = float("-inf")
        self.steps_without_improvement = 0

    def update(self, rewards_history: list) -> bool:
        if len(rewards_history) < self.window:
            return False
        current_avg = float(np.mean(rewards_history[-self.window:]))
        if current_avg > self.best + self.min_delta:
            self.best = current_avg
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        return self.steps_without_improvement >= self.patience

    def reset(self):
        self.best = float("-inf")
        self.steps_without_improvement = 0
