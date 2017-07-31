import gc, time  # noqa
import gym, gym_fast_envs  # noqa
import torch, numpy  # noqa

import utils
from utils import Preprocessor
from utils.VisdomMonitor import EvaluationMonitor
from agents import get_agent


def train_agent(cmdl):
    step_cnt = 0
    ep_cnt = 0
    start_time = time.time()

    env = utils.get_new_env(cmdl.env_name, cmdl)
    eval_env = EvaluationMonitor(gym.make(cmdl.env_name), cmdl)

    name = cmdl.agent.name
    agent = get_agent(name)(env.action_space, cmdl.agent)
    eval_agent = get_agent(name)(eval_env.action_space, cmdl.agent, False)

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

        if step_cnt % cmdl.report_freq == 0:
            agent.display_stats(start_time)
            agent.display_model_stats()
            gc.collect()

        if step_cnt % cmdl.eval_freq == 0:
            evaluate_agent(step_cnt, eval_env, eval_agent, agent.policy, cmdl)

    end_time = time.time()
    agent.display_final_report(ep_cnt, step_cnt, end_time - start_time)


def evaluate_agent(crt_training_step, eval_env, eval_agent, policy, cmdl):
    print("[Evaluator]  Initializing at %d training steps:" %
          crt_training_step)
    agent = eval_agent

    eval_env.get_crt_step(crt_training_step)
    agent.policy_evaluation.policy.load_state_dict(policy.state_dict())
    preprocess = Preprocessor(cmdl.env_class).transform

    step_cnt = 0
    o, r, done = eval_env.reset(), 0, False
    while step_cnt < cmdl.evaluator.eval_steps:
        s = preprocess(o)
        a = agent.evaluate_policy(s)
        o, r, done, _ = eval_env.step(a)
        step_cnt += 1
        if done:
            o, r, done = eval_env.reset(), 0, False


if __name__ == "__main__":

    # Parse cmdl args for the config file and return config as Namespace
    config = utils.parse_config_file(utils.parse_cmd_args())

    # Assuming everything in the config is deterministic already.
    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)

    # Let's do this!
    train_agent(config)
