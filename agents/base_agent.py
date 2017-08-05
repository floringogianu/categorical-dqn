import time
from termcolor import colored as clr
from utils import not_implemented


class BaseAgent(object):
    def __init__(self, action_space, is_training):
        self.actions = action_space
        self.action_no = self.actions.n

        self.is_training = is_training
        self.step_cnt = 0
        self.ep_cnt = 0
        self.ep_reward_cnt = 0
        self.ep_reward = []
        self.max_mean_rw = -100
        if is_training:
            print("Agent in training mode")
        else:
            print("Agent in evaluation mode")

    def evaluate_policy(self, obs):
        not_implemented(self)

    def improve_policy(self, _state, _action, reward, state, done):
        not_implemented(self)

    def gather_stats(self, reward, done):
        self.step_cnt += 1
        self.ep_reward_cnt += reward
        if done:
            self.ep_cnt += 1
            self.ep_reward.append(self.ep_reward_cnt)
            self.ep_reward_cnt = 0

    def display_setup(self, env, config):
        emph = ["env_name", "agent_type", "label", "batch_size", "lr"]
        print("-------------------------------------------------")
        for k in config.__dict__:
            if config.__dict__[k] is not None:
                v = config.__dict__[k]
                space = "." * (32 - len(k))
                config_line = "%s:  %s  %s" % (k, space, v)
                for e in emph:
                    if k == e:
                        config_line = clr(config_line, attrs=['bold'])
                print(config_line)
        print("-------------------------------------------------")
        custom = {"no_of_actions": self.action_no}
        for k, v in custom.items():
            space = "." * (32 - len(k))
            print("%s:  %s  %s" % (k, space, v))
        print("-------------------------------------------------")

    def display_stats(self, start_time):
        fps = self.cmdl.report_frequency / (time.perf_counter() - start_time)

        print(clr("[%s] step=%7d, fps=%.2f " % (self.name, self.step_cnt, fps),
                  attrs=['bold']))
        self.ep_reward.clear()

    def display_final_report(self, ep_cnt, step_cnt, elapsed_time):
        fps = step_cnt / elapsed_time
        print(clr("[  %s   ] finished after %d eps, %d steps."
              % ("Main", ep_cnt, step_cnt), 'white', 'on_magenta'))
        print(clr("[  %s   ] finished after %.2fs, %.2ffps."
              % ("Main", elapsed_time, fps), 'white', 'on_magenta'))

    def display_model_stats(self):
        pass
