import time
import torch
from utils import TorchTypes


class TensorCircularBuffer(object):
    def __init__(self, capacity, hist_len, state_dims, cuda):
        self.capacity = capacity
        self.state_dims = state_dims
        self.dtype = TorchTypes(cuda)

        self.position = 0
        # we won't be initializing the full memory on the CPU by default
        self.memory = {
            "_state": torch.ByteTensor(capacity, 1, *state_dims).fill_(0),
            "_action": torch.LongTensor(capacity, 1).fill_(0),
            "reward": torch.FloatTensor(capacity, 1).fill_(0),
            "done": torch.ByteTensor(capacity, 1).fill_(0)
        }
        self.full_idx = -1
        print("[Experience Replay]  Done allocating main memory.")
        time.sleep(10)

    def push(self, _s, _a, r, d):
        idx = self.position
        _s = _s.unsqueeze(0)  # (24, 24) -> (1, 24, 24)
        self.memory["_state"][idx] = _s * 255
        self.memory["_action"][idx, 0] = _a
        self.memory["reward"][idx, 0] = r
        self.memory["done"][idx, 0] = 0 if d else 1

        self.position = (self.position + 1) % self.capacity
        if self.full_idx < (self.capacity - 2):
            self.full_idx += 1

    def __len__(self):
        return self.full_idx


class TensorExperienceReplay(TensorCircularBuffer):
    def __init__(self, capacity, batch_size, hist_len, state_dims, cuda):
        TensorCircularBuffer.__init__(self, capacity, hist_len, state_dims,
                                      cuda)
        self.hist_len = hist_len
        self.batch_size = batch_size
        batch_state_dims = (batch_size, hist_len, *state_dims)
        dtype = self.dtype

        self._states = dtype.FT(*batch_state_dims).fill_(0)
        self._actions = dtype.LT(batch_size, 1).fill_(0)
        self.states = dtype.FT(*batch_state_dims).fill_(0)
        self.rewards = dtype.FT(batch_size, 1).fill_(0)
        self.done = dtype.BT(batch_size, 1).fill_(0)
        print("[Experience Replay]  Done allocating cuda batch.")

    def sample(self):
        batch_sz = self.batch_size
        memory = self.memory
        h = self.hist_len

        idxs = torch.LongTensor(batch_sz).random_(h, self.full_idx - 1)

        # need to figure out how to use idx directly
        for i in range(batch_sz):
            idx = idxs[i]
            self._states[i] = memory["_state"][idx-h:idx].float() / 255
            self.states[i] = memory["_state"][(idx-h)+1:idx+1].float() / 255
            self._actions[i] = memory["_action"][idx-1]
            self.rewards[i] = memory["reward"][idx-1]
            self.done[i] = memory["done"][idx-1]

        return [batch_sz, self._states, self._actions,
                self.rewards, self.states, self.done]

    """
    # This ain't faster.
    # Need to find a better solution for indexing
    def sample(self):
        batch_sz = self.batch_size
        memory = self.memory
        h = self.hist_len
        dtype = self.dtype

        idxs = list(torch.LongTensor(batch_sz).random_(h, self.full_idx - 1))

        s_idxs = [ix - j for ix in idxs for j in range(h)]
        ns_idxs = [(ix+1) - j for ix in idxs for j in range(h)]

        stx = torch.LongTensor(s_idxs).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        nstx = torch.LongTensor(ns_idxs).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        idxs = torch.LongTensor(idxs).unsqueeze(1)

        stx = stx.expand(len(stx), 1, *self.state_dims)
        nstx = stx.expand(len(nstx), 1, *self.state_dims)

        _states = (memory["_state"].gather(0, stx).float() / 255).view(
                batch_sz, h, *self.state_dims).type(dtype.FT)
        states = (memory["_state"].gather(0, nstx).float() / 255).view(
                batch_sz, h, *self.state_dims).type(dtype.FT)
        _actions = memory["_action"].gather(0, idxs).type(dtype.LT)
        rewards = memory["reward"].gather(0, idxs).type(dtype.FT)
        done = memory["done"].gather(0, idxs).type(dtype.BT)

        return [batch_sz, _states, _actions, rewards, states, done]
    """
