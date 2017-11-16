from torch.autograd import Variable


class DeterministicPolicy(object):
    def __init__(self, policy):
        """Assumes policy returns an autograd.Variable"""

        self.name = "DP"
        self.policy = policy
        self.cuda = next(policy.parameters()).is_cuda

    def get_action(self, state):
        """ Takes best action based on estimated state-action values."""
        state = state.cuda() if self.cuda else state
        q_val, argmax_a = self.policy(
                Variable(state, volatile=True)).data.max(1)
        """
        result = self.policy(Variable(state_batch, volatile=True))
        print(result)
        """
        return (q_val[0], argmax_a[0])
