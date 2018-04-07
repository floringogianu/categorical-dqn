""" Categorical DQN policy improvement.
"""
import torch
from torch.autograd import Variable
# import torch.nn.functional as F
from termcolor import colored as clr
from utils import TorchTypes
from policy_improvement.optim_utils import optim_factory, lr_schedule


class CategoricalPolicyImprovement(object):
    """ Deep Q-Learning training method. """

    def __init__(self, policy, target_policy, cmdl):
        self.name = "Categorical-PI"
        self.cmdl = cmdl
        self.policy = policy
        self.target_policy = target_policy
        self.lr = cmdl.lr
        self.gamma = cmdl.gamma

        self.optimizer = optim_factory(self.policy.parameters(), cmdl)
        self.optimizer.zero_grad()
        self.lr_generator = lr_schedule(cmdl.lr, 0.00001, cmdl.training_steps)

        self.dtype = dtype = TorchTypes(cmdl.cuda)
        self.v_min, self.v_max = v_min, v_max = cmdl.v_min, cmdl.v_max
        self.atoms_no = atoms_no = cmdl.atoms_no
        self.support = torch.linspace(v_min, v_max, atoms_no)
        self.support = self.support.type(dtype.FT)
        self.delta_z = (cmdl.v_max - cmdl.v_min) / (cmdl.atoms_no - 1)
        self.m = torch.zeros(cmdl.batch_size, self.atoms_no).type(dtype.FT)

    def accumulate_gradient(self, batch_sz, states, actions, rewards,
                            next_states, mask):
        """ Compute the difference between the return distributions of Q(s,a)
            and TQ(s_,a).
        """
        states = Variable(states)
        actions = Variable(actions)
        next_states = Variable(next_states, volatile=True)

        # Compute probabilities of Q(s,a*)
        q_probs = self.policy(states)
        actions = actions.view(batch_sz, 1, 1)
        action_mask = actions.expand(batch_sz, 1, self.atoms_no)
        qa_probs = q_probs.gather(1, action_mask).squeeze()

        # Compute distribution of Q(s_,a)
        target_qa_probs = self._get_categorical(next_states, rewards, mask)

        # Compute the cross-entropy of phi(TZ(x_,a)) || Z(x,a)
        qa_probs = qa_probs.clamp(min=1e-3)  # Tudor's trick for avoiding nans
        loss = - torch.sum(target_qa_probs * torch.log(qa_probs))

        # Accumulate gradients
        loss.backward()

    def update_model(self):
        if self.cmdl.optim == "RMSprop":
            lr = next(self.lr_generator)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _get_categorical(self, next_states, rewards, mask):
        batch_sz = next_states.size(0)
        gamma = self.gamma

        # Compute probabilities p(x, a)
        probs = self.target_policy(next_states).data
        qs = torch.mul(probs, self.support.expand_as(probs))
        argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
        action_mask = argmax_a.expand(batch_sz, 1, self.atoms_no)
        qa_probs = probs.gather(1, action_mask).squeeze()

        # Mask gamma and reshape it torgether with rewards to fit p(x,a).
        rewards = rewards.expand_as(qa_probs)
        gamma = (mask.float() * gamma).expand_as(qa_probs)

        # Compute projection of the application of the Bellman operator.
        bellman_op = rewards + gamma * self.support.unsqueeze(0).expand_as(rewards)
        bellman_op = torch.clamp(bellman_op, self.v_min, self.v_max)

        # Compute categorical indices for distributing the probability
        m = self.m.fill_(0)
        b = (bellman_op - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms_no - 1)) * (l == u)] += 1

        # Distribute probability
        """
        for i in range(batch_sz):
            for j in range(self.atoms_no):
                uidx = u[i][j]
                lidx = l[i][j]
                m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
                m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
        for i in range(batch_sz):
            m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
            m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))

        """
        # Optimized by https://github.com/tudor-berariu
        offset = torch.linspace(0, ((batch_sz - 1) * self.atoms_no), batch_sz)\
            .type(self.dtype.LT)\
            .unsqueeze(1).expand(batch_sz, self.atoms_no)

        m.view(-1).index_add_(0, (l + offset).view(-1),
                              (qa_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1),
                              (qa_probs * (b - l.float())).view(-1))
        return Variable(m)

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
        if mask[0] == 0:
            r = reward_batch[0, 0]
            if r == 1.0:
                print(r)

    def _debug_states(self, state_batch, next_state_batch, mask):
        for i in range(24):
            for j in range(24):
                px = state_batch[0, 0, i, j]
                if px < 0.90:
                    print(clr("%.2f  " % px, 'magenta'), end="")
                else:
                    print(("%.2f  " % px), end="")
            print()
        for i in range(24):
            for j in range(24):
                px = next_state_batch[0, 0, i, j]
                if px < 0.90:
                    print(clr("%.2f  " % px, 'magenta'), end="")
                else:
                    print(clr("%.2f  " % px, 'white'), end="")
            print()
        if mask[0] == 0:
            print(clr("Done batch ............", 'magenta'))
        else:
            print(".......................")
