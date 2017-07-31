from termcolor import colored as clr
from utils.monitors import VisdomMonitor


def get_new_env(env_name, cmdl):
    """Configure the training environment and return an instance."""
    import logging
    import gym
    import gym_fast_envs  # noqa
    from gym.wrappers import Monitor

    # Undo the default logger and configure a new one.
    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Configure environment
    outdir = '/tmp/nec/%s-results' % cmdl.label
    env = gym.make(env_name)
    env = Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(cmdl.seed)
    return env


def get_config_info(env_name):
    """Utility for configuring the model.
    Returns types and info necessary for the initial setup.
    """
    import gym
    import gym_fast_envs  # noqa
    env = gym.make(env_name)
    o, a = env.observation_space, env.action_space
    env.close()
    return o, a


def not_implemented(obj):
    import inspect
    method_name = inspect.stack()[1][3]
    raise RuntimeError(
        clr(("%s.%s not implemented nor delegated." %
            (obj.name, method_name)), 'white', 'on_red'))
