import gc, time  # noqa
import gym, gym_fast_envs  # noqa
import torch, numpy  # noqa

import utils
from agents import get_agent


def train_agent(cmdl):
    global_time = time.perf_counter()

    env = utils.env_factory(cmdl, "training")
    eval_env = utils.env_factory(cmdl, "evaluation")

    name = cmdl.agent_type
    env_space = (env.action_space, env.observation_space)
    agent = get_agent(name)(env_space, cmdl)
    eval_env_space = (env.action_space, env.observation_space)
    eval_agent = get_agent("evaluation")(eval_env_space, cmdl)

    agent.display_setup(env, cmdl)

    ep_cnt = 1
    fps_time = time.perf_counter()

    s, r, done = env.reset(), 0, False

    for step_cnt in range(cmdl.training_steps):
        a = agent.evaluate_policy(s)
        _s, _a = s.clone(), a
        s, r, done, _ = env.step(a)
        agent.improve_policy(_s, _a, r, s, done)

        step_cnt += 1
        agent.gather_stats(r, done)

        # Do some reporting
        if step_cnt != 0 and step_cnt % cmdl.report_frequency == 0:
            agent.display_stats(fps_time)
            agent.display_model_stats()
            fps_time = time.perf_counter()
            gc.collect()

        # Start doing an evaluation
        eval_ready = step_cnt >= cmdl.eval_start
        if eval_ready and (step_cnt % cmdl.eval_frequency == 0):
            eval_time = time.perf_counter()
            evaluate_agent(step_cnt, eval_env, eval_agent,
                           agent.policy, cmdl)
            gc.collect()
            fps_time = fps_time + (time.perf_counter() - eval_time)

        if done:
            ep_cnt += 1
            s, r, done = env.reset(), 0, False

    agent.display_final_report(ep_cnt, step_cnt, global_time)


def evaluate_agent(crt_training_step, eval_env, eval_agent, policy, cmdl):
    print("[Evaluator] starting @ %d training steps:" % crt_training_step)
    agent = eval_agent

    eval_env.get_crt_step(crt_training_step)
    # need to change this
    agent.policy_evaluation.policy.load_state_dict(policy.state_dict())

    step_cnt = 0
    s, r, done = eval_env.reset(), 0, False
    while step_cnt < cmdl.eval_steps:
        a = agent.evaluate_policy(s)
        s, r, done, _ = eval_env.step(a)
        step_cnt += 1
        if done:
            s, r, done = eval_env.reset(), 0, False


if __name__ == "__main__":

    # Parse cmdl args for the config file and return config as Namespace
    config = utils.parse_config_file(utils.parse_cmd_args())

    # Assuming everything in the config is deterministic already.
    torch.manual_seed(config.seed)
    numpy.random.seed(config.seed)
    torch.set_num_threads(4)

    # Let's do this!
    train_agent(config)
