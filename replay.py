from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add2(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def add(self, state):
        """Save a transition"""
        self.memory.append(state)

    def sample(self, batch_size, sequence_len):
        return self.memory[sequence_len * -1:]

    def sample3(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def sample2(self, batch_size, sequence_len):
        last_idx = len(self.memory) - sequence_len
        batch = []

        for _ in range(batch_size):
            start_idx = random.randint(0, last_idx)
            sequence = [self.memory[i].state for i in range(start_idx, start_idx + sequence_len)]
            batch.append(sequence)
        
        return batch

    def __len__(self):
        return len(self.memory)