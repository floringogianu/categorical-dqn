from torch.autograd import Variable


class DeterministicPolicy(object):
    def __init__(self, estimator):
        """Assumes estimator returns an autograd.Variable"""

        self.name = "DP"
        self.estimator = estimator

    def get_action(self, state_batch):
        """ Takes best action based on estimated state-action values."""
        q_val, argmax_a = self.estimator(
                Variable(state_batch, volatile=True)).data.max(1)
        return (q_val[0, 0], argmax_a[0, 0])
