import torch
from numpy.random import uniform
from agents.base_agent import BaseAgent
from estimators import get_estimator as get_model
from policy_evaluation import DeterministicPolicy as DQNEvaluation
from policy_evaluation import get_schedule as get_epsilon_schedule
from policy_improvement import DQNPolicyImprovement as DQNImprovement
from data_structures import ReplayMemory, Transition
from utils import TorchTypes


class DQNAgent(BaseAgent):
    def __init__(self, action_space, cmdl):
        BaseAgent.__init__(self, action_space)
        self.name = "DQN_agent"
        self.cmdl = cmdl
        eps = self.cmdl.epsilon
        e_steps = self.cmdl.epsilon_steps

        self.policy = policy = get_model(cmdl.estimator, 1, cmdl.hist_len,
                                         self.action_no, cmdl.hidden_size)
        self.target = target = get_model(cmdl.estimator, 1, cmdl.hist_len,
                                         self.action_no, cmdl.hidden_size)
        if self.cmdl.cuda:
            self.policy.cuda()
            self.target.cuda()
        self.policy_evaluation = DQNEvaluation(policy)
        self.policy_improvement = DQNImprovement(policy, target, cmdl)
        self.exploration = get_epsilon_schedule("linear", eps, 0.05, e_steps)
        self.replay_memory = ReplayMemory(capacity=cmdl.experience_replay)

        self.dtype = TorchTypes(cmdl.cuda)
        self.max_q = -1000

    def evaluate_policy(self, state):
        self.epsilon = next(self.exploration)

        if self.epsilon < uniform():
            state = self._frame2torch(state)
            qval, action = self.policy_evaluation.get_action(state)
            # print(qval, action)
            self.max_q = max(qval, self.max_q)
            return action
        else:
            return self.actions.sample()

    def improve_policy(self, _s, _a, r, s, done):
        self.replay_memory.push(_s, _a, s, r, done)

        if len(self.replay_memory) < self.cmdl.batch_size:
            return

        if (self.step_cnt % self.cmdl.update_freq == 0) and (
                len(self.replay_memory) > self.cmdl.batch_size):
            # get batch of transitions
            transitions = self.replay_memory.sample(self.cmdl.batch_size)
            batch = self._batch2torch(transitions)
            # compute gradients
            self.policy_improvement.accumulate_gradient(*batch)
            self.policy_improvement.update_model()

        if self.step_cnt % self.cmdl.target_update_freq == 0:
            self.policy_improvement.update_target_net()

    def display_model_stats(self):
        self.policy_improvement.get_model_stats()
        print("MaxQ=%2.2f.  MemSz=%5d.  Epsilon=%.2f." % (
                self.max_q, len(self.replay_memory), self.epsilon))

    def _frame2torch(self, s):
        state = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        return state.type(self.dtype.FloatTensor)

    def _batch2torch(self, batch, batch_sz=None):
        """ List of Transitions to List of torch states, actions, rewards.
            From a batch of transitions (s0, a0, Rt)
            get a batch of the form state=(s0,s1...), action=(a1,a2...),
            Rt=(rt1,rt2...)
        """
        batch_sz = len(batch) if batch_sz is None else batch_sz
        batch = Transition(*zip(*batch))
        # print("[%s] Batch len=%d" % (self.name, batch_sz))

        states = [torch.from_numpy(s).unsqueeze(0) for s in batch.state]
        states_ = [torch.from_numpy(s).unsqueeze(0) for s in batch.state_]

        state_batch = torch.stack(states).type(self.dtype.FloatTensor)
        action_batch = self.dtype.LongTensor(batch.action)
        reward_batch = self.dtype.FloatTensor(batch.reward)
        next_state_batch = torch.stack(states_).type(self.dtype.FloatTensor)

        # Compute a mask for terminal next states
        # [True, False, False] -> [1, 0, 0]::ByteTensor
        mask = 1 - self.dtype.ByteTensor(batch.done)

        return [batch_sz, state_batch, action_batch, reward_batch,
                next_state_batch, mask]
