import torch
from torch.autograd import Variable
from utils import TorchTypes


class CategoricalPolicyEvaluation(object):
    def __init__(self, policy, cmdl):
        """Assumes policy returns an autograd.Variable"""
        self.name = "CP"
        self.cmdl = cmdl
        self.policy = policy

        self.dtype = dtype = TorchTypes(cmdl.cuda)
        self.support = torch.linspace(cmdl.v_min, cmdl.v_max, cmdl.atoms_no)
        self.support = self.support.type(dtype.FT)

    def get_action(self, state):
        """ Takes best action based on estimated state-action values."""
        state = state.type(self.dtype.FT)
        probs = self.policy(Variable(state, volatile=True)).data
        support = self.support.expand_as(probs)
        q_val, argmax_a = torch.mul(probs, support).squeeze().sum(1).max(0)
        return (q_val[0], argmax_a[0])
