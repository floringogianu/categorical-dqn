import logging
import torch
import numpy as np
from gym import Wrapper
from gym import ObservationWrapper
from gym import RewardWrapper
from visdom import Visdom
from PIL import Image
from termcolor import colored as clr
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SqueezeRewards(RewardWrapper):
    def __init__(self, env):
        super(SqueezeRewards, self).__init__(env)
        print("[RewardWrapper] for clamping rewards to -+1")

    def _reward(self, reward):
        return float(np.sign(reward))


class PreprocessFrames(ObservationWrapper):
    def __init__(self, env, env_type, hist_len, state_dims):
        super(PreprocessFrames, self).__init__(env)

        self.env_type = env_type
        self.state_dims = state_dims
        self.hist_len = hist_len
        self.env_wh = self.env.observation_space.shape[0:2]
        self.env_ch = self.env.observation_space.shape[2]
        self.wxh = self.env_wh[0] * self.env_wh[1]

        # need to find a better way
        if self.env_type == "atari":
            self._preprocess = self._atari_preprocess
        elif self.env_type == "catch":
            self._preprocess = self._catch_preprocess
        print("[Preprocess Wrapper] for %s with state history of %d frames."
              % (self.env_type, hist_len))

        self.rgb = torch.FloatTensor([.2126, .7152, .0722])

        # torch.size([1, 4, 24, 24])
        """
        self.hist_state = torch.FloatTensor(1, hist_len, *state_dims)
        self.hist_state.fill_(0)
        """

        self.d = OrderedDict({i: torch.FloatTensor(1, 1, *state_dims).fill_(0)
                              for i in range(hist_len)})

    def _observation(self, o):
        return self._preprocess(o)

    def _reset(self):
        # self.hist_state.fill_(0)
        self.d = OrderedDict(
            {i: torch.FloatTensor(1, 1, *self.state_dims).fill_(0)
                for i in range(self.hist_len)})
        observation = self.env.reset()
        return self._observation(observation)

    def _catch_preprocess(self, o):
        return self._get_concatenated_state(self._rgb2y(o))

    def _atari_preprocess(self, o):
        img = Image.fromarray(self._rgb2y(o).numpy())
        img = np.array(img.resize(self.state_dims, resample=Image.NEAREST))
        th_img = torch.from_numpy(img)
        return self._get_concatenated_state(th_img)

    def _rgb2y(self, o):
        o = torch.from_numpy(o).float()
        return o.view(self.wxh, 3).mv(self.rgb).view(*self.env_wh) / 255

    def _get_concatenated_state(self, o):
        hist_len = self.hist_len
        for i in range(hist_len - 1):
            self.d[i] = self.d[i + 1]
        self.d[hist_len - 1] = o.unsqueeze(0).unsqueeze(0)
        return torch.cat(list(self.d.values()), 1)

    """
    def _get_concatenated_state(self, o):
        hist_len = self.hist_len  # eg. 4

        # move frames already existent one position below
        if hist_len > 1:
            self.hist_state[0][0:hist_len - 1] = self.hist_state[0][1:hist_len]

        # concatenate the newest frame to the top of the augmented state
        self.hist_state[0][self.hist_len - 1] = o
        return self.hist_state
    """


class VisdomMonitor(Wrapper):
    def __init__(self, env, cmdl):
        super(VisdomMonitor, self).__init__(env)

        self.freq = cmdl.report_freq  # in steps
        self.cmdl = cmdl

        if self.cmdl.display_plots:
            self.vis = Visdom()
            self.plot = self.vis.line(
                Y=np.array([0]), X=np.array([0]),
                opts=dict(
                    title=cmdl.label,
                    caption="Episodic reward per 1200 steps.")
            )

        self.step_cnt = 0
        self.ep_cnt = -1
        self.ep_rw = []
        self.last_reported_ep = 0

    def _step(self, action):
        # self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def _reset(self):
        self._before_reset()
        observation = self.env.reset()
        self._after_reset(observation)
        return observation

    def _after_step(self, o, r, done, info):
        self.ep_rw[self.ep_cnt] += r
        self.step_cnt += 1
        if self.step_cnt % self.freq == 0:
            self._update_plot()
        return done

    def _before_reset(self):
        self.ep_rw.append(0)

    def _after_reset(self, observation):
        self.ep_cnt += 1
        # print("[%2d][%4d]  RESET" % (self.ep_cnt, self.step_cnt))

    def _update_plot(self):
        # print(self.last_reported_ep, self.ep_cnt + 1)
        completed_eps = self.ep_rw[self.last_reported_ep:self.ep_cnt + 1]
        ep_mean_reward = sum(completed_eps) / len(completed_eps)
        if self.cmdl.display_plots:
            self.vis.line(
                X=np.array([self.step_cnt]),
                Y=np.array([ep_mean_reward]),
                win=self.plot,
                update='append'
            )
        self.last_reported_ep = self.ep_cnt + 1


class EvaluationMonitor(Wrapper):
    def __init__(self, env, cmdl):
        super(EvaluationMonitor, self).__init__(env)

        self.freq = cmdl.eval_frequency  # in steps
        self.eval_steps = cmdl.eval_steps
        self.cmdl = cmdl

        if self.cmdl.display_plots:
            self.vis = Visdom()
            self.plot = self.vis.line(
                Y=np.array([0]), X=np.array([0]),
                opts=dict(
                    title=cmdl.label,
                    caption="Episodic reward per %d steps." % self.eval_steps)
            )

        self.crt_step = 0
        self.step_cnt = 0
        self.ep_cnt = 0
        self.total_rw = 0
        self.max_mean_rw = -100

    def get_crt_step(self, crt_step):
        self.crt_step = crt_step

    def _reset_monitor(self):
        self.step_cnt, self.ep_cnt, self.total_rw = 0, 0, 0

    def _step(self, action):
        # self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def _reset(self):
        observation = self.env.reset()
        self._after_reset(observation)
        return observation

    def _after_step(self, o, r, done, info):
        self.total_rw += r
        self.step_cnt += 1
        if self.step_cnt == self.eval_steps:
            self._update_plot()
            self._reset_monitor()
        return done

    def _after_reset(self, observation):
        self.ep_cnt += 1
        # print("[%2d][%4d]  RESET" % (self.ep_cnt, self.step_cnt))

    def _update_plot(self):
        mean_rw = self.total_rw / self.ep_cnt
        max_mean_rw = self.max_mean_rw
        bg_color = 'on_blue'
        bg_color = 'on_magenta' if mean_rw > max_mean_rw else bg_color
        self.max_mean_rw = mean_rw if mean_rw > max_mean_rw else max_mean_rw

        if self.cmdl.display_plots:
            self.vis.line(
                X=np.array([self.crt_step]),
                Y=np.array([mean_rw]),
                win=self.plot,
                update='append'
            )
        print(clr("[Evaluator] done in %5d steps. " % self.step_cnt,
              attrs=['bold'])
              + clr(" rw/ep=%3.2f " % mean_rw, 'white', bg_color,
                    attrs=['bold']))


if __name__ == "__main__":
    import gym
    import gym_fast_envs  # noqa

    env = gym.make("Catcher-Level0-v0")
    # env = VisdomMonitor(env, 48)
    env = PreprocessFrames(env, "catch", 2, (24, 24))

    step_cnt = 0
    for e in range(2):
        o, r, done = env.reset(), 0, False
        print("----------------------------------------------------------")
        print(torch.sum(o, 3))
        print("**********************************************************")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Episode: ", e)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        while not done:
            o, r, done, _ = env.step(env.action_space.sample())
            print("----------------------------------------------------------")
            print(torch.sum(o, 3))
            print("**********************************************************")
            step_cnt += 1
