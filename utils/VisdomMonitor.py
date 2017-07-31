import logging
from gym import Wrapper
from visdom import Visdom
import numpy as np
from termcolor import colored as clr

logger = logging.getLogger(__name__)


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

        self.freq = cmdl.eval_freq  # in steps
        self.eval_steps = cmdl.evaluator.eval_steps
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
              'grey', 'on_white')
              + clr(" rw/ep=%3.2f " % mean_rw, 'white', bg_color))


if __name__ == "__main__":
    import gym
    import gym_fast_envs  # noqa

    env = gym.make("Catcher-Level0-v0")
    env = VisdomMonitor(env, 48)

    step_cnt = 0
    for e in range(8):
        o, r, done = env.reset(), 0, False
        while not done:
            o, r, done, _ = env.step(env.action_space.sample())
            step_cnt += 1
