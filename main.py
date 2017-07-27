import gc
import torch
import numpy
import time

import utils
from utils import Preprocessor
from agents import get_agent


def train_agent(cmdl):
    step_cnt = 0
    ep_cnt = 0
    start_time = time.time()

    env = utils.get_new_env(cmdl.env_name)
    agent = get_agent(cmdl.agent.name)(env.action_space, cmdl.agent)

    preprocess = Preprocessor(cmdl.env_class).transform
    agent.display_setup(env, cmdl)

    while step_cnt < cmdl.training.step_no:

        ep_cnt += 1
        o, r, done = env.reset(), 0, False
        s = preprocess(o)

        while not done:
            a = agent.evaluate_policy(s)
            o, r, done, _ = env.step(a)
            _s, _a = s, a
            s = preprocess(o)
            agent.improve_policy(_s, _a, r, s, done)

            step_cnt += 1
            agent.gather_stats(r, done)

        if ep_cnt % cmdl.report_freq == 0:
            agent.display_stats(start_time)
            agent.display_model_stats()
            gc.collect()

    end_time = time.time()
    agent.display_final_report(ep_cnt, step_cnt, end_time - start_time)


if __name__ == "__main__":

    # Parse cmdl args for the config file and return config as Namespace
    config = utils.parse_config_file(utils.parse_cmd_args())

    # Assuming everything in the config is deterministic already.
    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)

    # Let's do this!
    train_agent(config)
