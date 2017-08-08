import logging
import gym
import gym_fast_envs  # noqa
from termcolor import colored as clr

from utils import EvaluationMonitor
from utils import PreprocessFrames
from utils import SqueezeRewards


def env_factory(cmdl, mode):
    # Undo the default logger and configure a new one.
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    print(clr("[Main] Constructing %s environment." % mode, attrs=['bold']))
    env = gym.make(cmdl.env_name)

    if hasattr(cmdl, 'rescale_dims'):
        state_dims = (cmdl.rescale_dims, cmdl.rescale_dims)
    else:
        state_dims = env.observation_space.shape[0:2]

    if mode == "training":
        env = PreprocessFrames(env, cmdl.env_class, cmdl.hist_len, state_dims)
        if hasattr(cmdl, 'reward_clamp') and cmdl.reward_clamp:
            env = SqueezeRewards(env)
        print('-' * 50)
        return env

    elif mode == "evaluation":
        if cmdl.eval_env_name != cmdl.env_name:
            print(clr("[%s] Warning! evaluating on a different env: %s"
                      % ("Main", cmdl.eval_env_name), 'red', attrs=['bold']))
            env = gym.make(cmdl.eval_env_name)

        env = PreprocessFrames(env, cmdl.env_class, cmdl.hist_len, state_dims)
        env = EvaluationMonitor(env, cmdl)
        print('-' * 50)
        return env


def not_implemented(obj):
    import inspect
    method_name = inspect.stack()[1][3]
    raise RuntimeError(
        clr(("%s.%s not implemented nor delegated." %
            (obj.name, method_name)), 'white', 'on_red'))
