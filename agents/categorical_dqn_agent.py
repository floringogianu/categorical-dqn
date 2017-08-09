from agents.dqn_agent import DQNAgent
from estimators import get_estimator as get_model
from policy_evaluation import CategoricalPolicyEvaluation
from policy_improvement import CategoricalPolicyImprovement


class CategoricalDQNAgent(DQNAgent):
    def __init__(self, action_space, cmdl):
        DQNAgent.__init__(self, action_space, cmdl)
        self.name = "Categorical_agent"
        self.cmdl = cmdl

        hist_len, action_no = cmdl.hist_len, self.action_no
        self.policy = policy = get_model(cmdl.estimator, 1, hist_len,
                                         (action_no, cmdl.atoms_no),
                                         hidden_size=cmdl.hidden_size)
        self.target = target = get_model(cmdl.estimator, 1, hist_len,
                                         (action_no, cmdl.atoms_no),
                                         hidden_size=cmdl.hidden_size)
        if self.cmdl.cuda:
            self.policy.cuda()
            self.target.cuda()

        self.policy_evaluation = CategoricalPolicyEvaluation(policy, cmdl)
        self.policy_improvement = CategoricalPolicyImprovement(
                policy, target, cmdl)

    def improve_policy(self, _s, _a, r, s, done):
        h = self.cmdl.hist_len - 1
        self.replay_memory.push(_s[0, h], _a, r, done)

        if len(self.replay_memory) < self.cmdl.batch_size:
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
