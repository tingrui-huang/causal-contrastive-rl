import random


class ReplayBuffer:
    def __init__(self, capacity=10000, seed=None):
        self.capacity = capacity
        self.storage = []
        self.ptr = 0
        self.rng = random.Random(seed)

    def add(self, item):
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        return self.rng.sample(self.storage, batch_size)

    def __len__(self):
        return len(self.storage)