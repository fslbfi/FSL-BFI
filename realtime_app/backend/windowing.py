import numpy as np
from collections import deque

from realtime_app.config import DEFAULTS


class WindowBuffer:
    def __init__(self, window_size=None):
        self.window_size = window_size or DEFAULTS['window_size']
        self.buffer = deque(maxlen=self.window_size)

    def push(self, vmatrix):
        self.buffer.append(vmatrix)

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        if not self.is_ready():
            return None
        return np.stack(list(self.buffer), axis=0)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
