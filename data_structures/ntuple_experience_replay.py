import torch
from collections import namedtuple
from utils import TorchTypes


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done'))
BatchTransition = namedtuple('BatchTransition',
                             ('state', 'action', 'reward', 'state_', 'done'))


class CircularBuffer(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.fill_idx = -1

    def push(self, s, a, r, d):
        s = s.unsqueeze(0).unsqueeze(0)  # [24 24] --> [1, 1, 24, 24]
        if len(self.memory) < self.capacity:
            self.memory.append(Transition((s * 255).byte(), a, r, d))
            self.fill_idx += 1
        else:
            self.memory[self.position] = Transition((s * 255).byte(), a, r, d)
        self.position = (self.position + 1) % self.capacity

    def get_batch(self):
        return self.memory[:self.position]

    def reset(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


class nTupleExperienceReplay(CircularBuffer):
    def __init__(self, capacity, batch_size, hist_len, cuda):
        CircularBuffer.__init__(self, capacity)
        self.batch_size = batch_size
        self.dtype = TorchTypes(cuda)
        self.hist_len = hist_len

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return self._sample(batch_size)

    def _sample(self, batch_size=None):
        fidx = self.fill_idx - 1  # we can only index up to capacity - 2
        hist_len = self.hist_len
        mem = self.memory

        # sample batch_size indices
        idxs = torch.LongTensor(batch_size).random_(hist_len, fidx)

        # retrieve a list of ((hist_len + 1 transitions) * batch_size)
        samples = [mem[idxs[i]-hist_len:idxs[i]+1] for i in range(batch_size)]

        # concatenate frames for s and s_
        # and create a new list of transitions (s, a, r_, s_, d_)
        transitions = [BatchTransition(
            torch.cat([samples[j][i].state for i in range(hist_len)], 1),
            samples[j][hist_len-1].action,  # after idx of s
            samples[j][hist_len-1].reward,  # after idx of s
            torch.cat([samples[j][i].state for i in range(1, hist_len+1)], 1),
            samples[j][hist_len-1].done) for j in range(batch_size)]

        return self._batch2torch(transitions, self.batch_size)

    def _batch2torch(self, batch, batch_size):
        """ List of transitions -> Batch of transitions -> pytorch tensors.

            Returns:
                states: torch.size([batch_size, hist_len, w, h])
                a/r/d: torch.size([batch_size, 1])
        """
        # check-out pytorch dqn tutorial.
        # (t1, t2, ... tn) -> t((s1, s2, ..., sn), (a1, a2, ... an) ...)
        batch = BatchTransition(*zip(*batch))

        # lists to tensors
        state_batch = torch.cat(batch.state, 0).type(self.dtype.FT) / 255
        action_batch = self.dtype.LT(batch.action).unsqueeze(1)
        reward_batch = self.dtype.FT(batch.reward).unsqueeze(1)
        next_state_batch = torch.cat(batch.state_, 0).type(self.dtype.FT) / 255
        # [False, False, True, False] -> [1, 1, 0, 1]::ByteTensor
        mask = 1 - self.dtype.BT(batch.done).unsqueeze(1)

        return [batch_size, state_batch, action_batch, reward_batch,
                next_state_batch, mask]


class CachedExperienceReplay(nTupleExperienceReplay):
    def __init__(self, capacity, batch_size, hist_len, cuda, cached_batches):
        nTupleExperienceReplay.__init__(self, capacity, batch_size, hist_len,
                                        cuda)

        self.cached_batches = cached_batches  # no of cached batches
        self.cache_size = cached_batches * batch_size
        self.sample_idx = 0

    def sample(self):
        if self.sample_idx % self.cached_batches == 0:
            self._fill_cache()
            self.sample_idx = 0
        cache = self._sample_from_cache(self.sample_idx)
        self.sample_idx += 1
        return cache

    def _fill_cache(self):
        sz = self.cache_size
        _, self.cs, self.ca, self.cr, self.cns, self.cd = self._sample(sz)

    def _sample_from_cache(self, batch_idx):
        batch_sz = self.batch_size
        sidx = batch_sz * batch_idx
        eidx = sidx + batch_sz
        return [
            batch_sz,
            self.cs[sidx:eidx],
            self.ca[sidx:eidx],
            self.cr[sidx:eidx],
            self.cns[sidx:eidx],
            self.cd[sidx:eidx]
        ]
