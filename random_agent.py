from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, action_space, cmdl):
        BaseAgent.__init__(self, action_space)

        self.name = "RND_agent"

    def evaluate_policy(self, state):
        return self.action_space.sample()

    def improve_policy(self, _state, _action, reward, state, done):
        pass
