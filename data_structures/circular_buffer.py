from .transition import Transition


class CircularBuffer(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        self.position = self.position % self.capacity

    def get_batch(self):
        return self.memory[:self.position]

    def reset(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)
