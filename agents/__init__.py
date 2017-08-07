# from .nec_agent import NECAgent
from agents.dqn_agent import DQNAgent
from agents.categorical_dqn_agent import CategoricalDQNAgent
from agents.random_agent import RandomAgent
from agents.evaluation_agent import EvaluationAgent

AGENTS = {
    # "nec": NECAgent,
    "evaluation": EvaluationAgent,
    "categorical": CategoricalDQNAgent,
    "dqn": DQNAgent,
    "random": RandomAgent
}


def get_agent(name):
    return AGENTS[name]
