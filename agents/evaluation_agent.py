from numpy.random import uniform
from estimators import get_estimator as get_model
from policy_evaluation import DeterministicPolicy
from policy_evaluation import CategoricalPolicyEvaluation


class EvaluationAgent(object):
    def __init__(self, env_space, cmdl):
        self.name = "Evaluation"

        self.actions = env_space[0]
        self.action_no = action_no = self.actions.n
        self.cmdl = cmdl
        self.epsilon = 0.05

        if cmdl.agent_type == "dqn":
            self.policy = policy = get_model(cmdl.estimator, 1, cmdl.hist_len,
                                             self.action_no, cmdl.hidden_size)
            if self.cmdl.cuda:
                self.policy.cuda()
            self.policy_evaluation = DeterministicPolicy(policy)
        elif cmdl.agent_type == "categorical":
            self.policy = policy = get_model(cmdl.estimator, 1, cmdl.hist_len,
                                             (action_no, cmdl.atoms_no),
                                             hidden_size=cmdl.hidden_size)
            if self.cmdl.cuda:
                self.policy.cuda()
            self.policy_evaluation = CategoricalPolicyEvaluation(policy, cmdl)
        print("[%s]  Evaluating %s agent." % (self.name, cmdl.agent_type))

        self.max_q = -1000

    def evaluate_policy(self, state):
        if self.epsilon < uniform():
            qval, action = self.policy_evaluation.get_action(state)
            self.max_q = max(qval, self.max_q)
            return action
        else:
            return self.actions.sample()
