""" Deep Q-Learning policy improvement.
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from termcolor import colored as clr
from utils import TorchTypes
from policy_improvement.optim_utils import optim_factory, lr_schedule


class DQNPolicyImprovement(object):
    """ Deep Q-Learning training method. """

    def __init__(self, policy, target_policy, cmdl):
        self.name = "DQN-PI"
        self.cmdl = cmdl
        self.policy = policy
        self.target_policy = target_policy
        self.lr = cmdl.lr
        self.gamma = cmdl.gamma

        self.optimizer = optim_factory(self.policy.parameters(), cmdl)
        self.optimizer.zero_grad()
        self.lr_generator = lr_schedule(cmdl.lr, 0.00001, cmdl.training_steps)

        self.dtype = TorchTypes(cmdl.cuda)

    def accumulate_gradient(self, batch_sz, states, actions, rewards,
                            next_states, mask):
        """ Compute the temporal difference error.
            td_error = (r + gamma * max Q(s_,a)) - Q(s,a)
        """
        states = Variable(states)
        actions = Variable(actions)
        rewards = Variable(rewards.squeeze())
        next_states = Variable(next_states, volatile=True)

        # Compute Q(s, a)
        q_values = self.policy(states)
        q_values = q_values.gather(1, actions)

        # Compute Q(s_, a)
        q_target_values = Variable(torch.zeros(batch_sz).type(self.dtype.FT))

        # Bootstrap for non-terminal states
        q_target_values[mask] = self.target_policy(next_states).max(1)[0][mask]
        q_target_values.volatile = False      # So we don't mess the huber loss
        expected_q_values = (q_target_values * self.gamma) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)

        # Accumulate gradients
        loss.backward()

    def update_model(self):
        if self.cmdl.optim == "RMSprop":
            lr = next(self.lr_generator)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_net(self):
        """ Update the target net with the parameters in the online model."""
        self.target_policy.load_state_dict(self.policy.state_dict())

    def get_model_stats(self):
        param_abs_mean = 0
        grad_abs_mean = 0
        t_param_abs_mean = 0
        n_params = 0
        for p in self.policy.parameters():
            param_abs_mean += p.data.abs().sum()
            grad_abs_mean += p.grad.data.abs().sum()
            n_params += p.data.nelement()
        for t in self.target_policy.parameters():
            t_param_abs_mean += t.data.abs().sum()

        print("Wm: %.9f | Gm: %.9f | Tm: %.9f" % (
            param_abs_mean / n_params,
            grad_abs_mean / n_params,
            t_param_abs_mean / n_params))

    def _debug_transitions(self, mask, reward_batch):
        if mask[0, 0] == 0:
            r = reward_batch.data[0, 0]
            if r == 1.0:
                print(r)

    def _debug_states(self, state_batch, next_state_batch, mask, target):
        batch_idx = 23
        for k in range(state_batch.size(1)):
            for i in range(24):
                for j in range(24):
                    px = state_batch[batch_idx, k, i, j]
                    if px < 0.90:
                        print(clr("%.2f  " % px, 'magenta'), end="")
                    else:
                        print(("%.2f  " % px), end="")
                print()
            print()
        print("************ NEXT STATE *********************")
        for v in range(next_state_batch.size(1)):
            for i in range(24):
                for j in range(24):
                    px = next_state_batch[batch_idx, v, i, j]
                    if px < 0.90:
                        print(clr("%.2f  " % px, 'magenta'), end="")
                    else:
                        print(clr("%.2f  " % px, 'white'), end="")
                print()
            print()
        if mask[batch_idx, 0] == 0:
            print(clr("Done batch ............", 'magenta'))
            print(target[batch_idx])
        else:
            print(".......................")
