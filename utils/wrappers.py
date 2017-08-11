import unittest
import logging
import torch
import numpy as np
import gym
from gym import Wrapper
from gym import ObservationWrapper
from gym import RewardWrapper
from PIL import Image
from termcolor import colored as clr
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SqueezeRewards(RewardWrapper):
    def __init__(self, env):
        super(SqueezeRewards, self).__init__(env)
        print("[Reward Wrapper] for clamping rewards to -+1")

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


class DoneAfterLostLife(gym.Wrapper):
    def __init__(self, env):
        super(DoneAfterLostLife, self).__init__(env)

        self.no_more_lives = True
        self.crt_live = env.unwrapped.ale.lives()
        self.has_many_lives = self.crt_live != 0

        if self.has_many_lives:
            self._step = self._many_lives_step
        else:
            self._step = self._one_live_step
        not_a = clr("not a", attrs=['bold'])

        print("[DoneAfterLostLife Wrapper]  %s is %s many lives game."
              % (env.env.spec.id, "a" if self.has_many_lives else not_a))

    def _reset(self):
        if self.no_more_lives:
            obs = self.env.reset()
            self.crt_live = self.env.unwrapped.ale.lives()
            return obs
        else:
            return self.__obs

    def _many_lives_step(self, action):
        obs, reward, done, info = self.env.step(action)
        crt_live = self.env.unwrapped.ale.lives()
        if crt_live < self.crt_live:
            # just lost a live
            done = True
            self.crt_live = crt_live

        if crt_live == 0:
            self.no_more_lives = True
        else:
            self.no_more_lives = False
            self.__obs = obs
        return obs, reward, done, info

    def _one_live_step(self, action):
        return self.env.step(action)


class EvaluationMonitor(Wrapper):
    def __init__(self, env, cmdl):
        super(EvaluationMonitor, self).__init__(env)

        self.freq = cmdl.eval_frequency  # in steps
        self.eval_steps = cmdl.eval_steps
        self.cmdl = cmdl

        if self.cmdl.display_plots:
            import Visdom
            self.vis = Visdom()
            self.plot = self.vis.line(
                Y=np.array([0]), X=np.array([0]),
                opts=dict(
                    title=cmdl.label,
                    caption="Episodic reward per %d steps." % self.eval_steps)
            )

        self.eval_cnt = 0
        self.crt_training_step = 0
        self.step_cnt = 0
        self.ep_cnt = 1
        self.total_rw = 0
        self.max_mean_rw = -1000

        no_of_evals = cmdl.training_steps // cmdl.eval_frequency \
            - (cmdl.eval_start-1) // cmdl.eval_frequency

        self.eval_frame_idx = torch.LongTensor(no_of_evals).fill_(0)
        self.eval_rw_per_episode = torch.FloatTensor(no_of_evals).fill_(0)
        self.eval_rw_per_frame = torch.FloatTensor(no_of_evals).fill_(0)
        self.eval_eps_per_eval = torch.LongTensor(no_of_evals).fill_(0)

    def get_crt_step(self, crt_training_step):
        self.crt_training_step = crt_training_step

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

        # Evaluation ends here
        if self.step_cnt == self.eval_steps:
            self._update()
            self._reset_monitor()
        return done

    def _after_reset(self, observation):
        if self.step_cnt != self.eval_steps:
            self.ep_cnt += 1

    def _update(self):
        mean_rw = self.total_rw / (self.ep_cnt - 1)
        max_mean_rw = self.max_mean_rw
        self.max_mean_rw = mean_rw if mean_rw > max_mean_rw else max_mean_rw

        self._update_plot(self.crt_training_step, mean_rw)
        self._display_logs(mean_rw, max_mean_rw)
        self._update_reports(mean_rw)
        self.eval_cnt += 1

    def _update_reports(self, mean_rw):
        idx = self.eval_cnt

        self.eval_frame_idx[idx] = self.crt_training_step
        self.eval_rw_per_episode[idx] = mean_rw
        self.eval_rw_per_frame[idx] = self.total_rw / self.step_cnt
        self.eval_eps_per_eval[idx] = (self.ep_cnt - 1)

        torch.save({
            'eval_frame_idx': self.eval_frame_idx,
            'eval_rw_per_episode': self.eval_rw_per_episode,
            'eval_rw_per_frame': self.eval_rw_per_frame,
            'eval_eps_per_eval': self.eval_eps_per_eval
        }, self.cmdl.results_path + "/eval_stats.torch")

    def _update_plot(self, crt_training_step, mean_rw):
        if self.cmdl.display_plots:
            self.vis.line(
                X=np.array([crt_training_step]),
                Y=np.array([mean_rw]),
                win=self.plot,
                update='append'
            )

    def _display_logs(self, mean_rw, max_mean_rw):
        bg_color = 'on_magenta' if mean_rw > max_mean_rw else 'on_blue'
        print(clr("[Evaluator] done in %5d steps. " % self.step_cnt,
              attrs=['bold'])
              + clr(" rw/ep=%3.2f " % mean_rw, 'white', bg_color,
                    attrs=['bold']))


class VisdomMonitor(Wrapper):
    def __init__(self, env, cmdl):
        super(VisdomMonitor, self).__init__(env)

        self.freq = cmdl.report_freq  # in steps
        self.cmdl = cmdl

        if self.cmdl.display_plots:
            from visdom import Visdom
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


class TestAtariWrappers(unittest.TestCase):

    def _test_env(self, env_name):
        env = gym.make(env_name)
        env = DoneAfterLostLife(env)

        o = env.reset()

        for i in range(10000):
            o, r, d, _ = env.step(env.action_space.sample())
            if d:
                o = env.reset()
                print("%3d, %s, %d" % (i, env_name, env.unwrapped.ale.lives()))

    def test_pong(self):
        print("Testing Pong")
        self._test_env("Pong-v0")

    def test_frostbite(self):
        print("Testing Frostbite")
        self._test_env("Frostbite-v0")


if __name__ == "__main__":
    import unittest
    unittest.main()
