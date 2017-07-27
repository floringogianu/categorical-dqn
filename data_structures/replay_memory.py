"""Just a buffer for now
"""
import random
from .circular_buffer import CircularBuffer


class ReplayMemory(CircularBuffer):
    def __init__(self, capacity):
        CircularBuffer.__init__(self, capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
