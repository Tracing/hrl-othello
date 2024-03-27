import collections
import heapq
import random
import numpy as np
import tensorflow as tf

class ReplayBuffer(collections.deque):
    def __init__(self, size):
        super(ReplayBuffer, self).__init__()
        self.size = size

    def store(self, x):
        self.append(x)
        if len(self) > self.size:
            self.popleft()

    def sample_minibatch(self, N):
        if len(self) == N:
            sample = list(self)
        else:
            sample = random.sample(self, N)
            
        n = len(sample[0])
        ret = []
    
        for i in range(n):
            xs = []
            for j in range(N):
                xs.append(sample[j][i])
            ret.append(tf.concat(xs, 0))
        return tuple(ret)