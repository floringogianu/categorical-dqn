from numpy.random import uniform
from agents.base_agent import BaseAgent
from estimators import get_estimator as get_model
from policy_evaluation import DeterministicPolicy as DQNEvaluation
from policy_evaluation import get_schedule as get_epsilon_schedule
from policy_improvement import DQNPolicyImprovement as DQNImprovement
from data_structures import ExperienceReplay
from utils import TorchTypes


class DQNAgent(BaseAgent):
    def __init__(self, env_space, cmdl):
        BaseAgent.__init__(self, env_space)
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
        self.replay_memory = ExperienceReplay.factory(cmdl, self.state_dims)

        self.dtype = TorchTypes(cmdl.cuda)
        self.max_q = -1000

    def evaluate_policy(self, state):
        self.epsilon = next(self.exploration)
        if self.epsilon < uniform():
            qval, action = self.policy_evaluation.get_action(state)
            self.max_q = max(qval, self.max_q)
            return action
        else:
            return self.actions.sample()

    def improve_policy(self, _s, _a, r, s, done):
        h = self.cmdl.hist_len - 1
        self.replay_memory.push(_s[0, h], _a, r, done)

        if len(self.replay_memory) < self.cmdl.start_learning_after:
            return

        if (self.step_cnt % self.cmdl.update_freq == 0) and (
                len(self.replay_memory) > self.cmdl.batch_size):

            # get batch of transitions
            batch = self.replay_memory.sample()

            # compute gradients
            self.policy_improvement.accumulate_gradient(*batch)
            self.policy_improvement.update_model()

        if self.step_cnt % self.cmdl.target_update_freq == 0:
            self.policy_improvement.update_target_net()

    def display_model_stats(self):
        self.policy_improvement.get_model_stats()
        print("MaxQ=%2.2f.  MemSz=%5d.  Epsilon=%.2f." % (
                self.max_q, len(self.replay_memory), self.epsilon))
        self.max_q = -1000
