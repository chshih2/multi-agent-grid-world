import sys

sys.path.append("../../")
import numpy as np
import gym
# from agent_simulator import AgentSimulator
from post_processing import plot_video
import matplotlib.pyplot as plt


class Environment(gym.Env):
    def __init__(self, task, env_config, seed):
        ''' General Setup '''
        self.rng = np.random.default_rng(seed)
        self.metadata = {}

        ''' Task '''
        self.task = task
        self.task.set_rng(self.rng)
        self.action_space = self.task.action_space
        self.observation_space = self.task.observation_space

        ''' Agent '''
        self.n_agent = env_config['num_agent']
        # self.agent_simulator = AgentSimulator(env_config, self.task)

        self.state = np.zeros(task.agent_state_dim + task.goal_dim)

        self.use_pretrained_policy = env_config['env_name'] == 'multi-agent'
        self.pretrained_policy_flag = False

    def reset(self):
        if not self.pretrained_policy_flag and self.use_pretrained_policy:
            self.task.set_pretrained_task()
            self.pretrained_policy_flag = True
        ''' Reset arms'''
        self.task.task_reset()
        # self.agent_simulator.reset(rod_start_list, self.task.ball_start, self.task.target_center)
        self.time = np.float64(0.0)
        # self.task.set_agent(self.agent_simulator)

        ''' Reset parameters '''
        self.state = self.task.state_fn()
        # print(self.task.grid)
        return self.state

    def step(self, action):
        # action (n_arm, action dim (+ task action dim))
        # print(self.state)
        self.task.action_fn(action)

        self.reward = self.task.reward_fn()
        # print(self.task.grid)
        # print(self.task.shooting_a_goal,self.reward)
        #
        # print("====")
        done = self.task.task_done()
        if done:
            self.state = None
        else:
            self.state = self.task.state_fn()

        return self.state, self.reward, done, {'last_value': self.task.last_value,
                                               'success': self.task.task_complete}

    def save_data(self, filename):
        filename_video = filename + ".mp4"
        plot_video(self.task.render_agent, self.task.render_ball, self.task.render_target,
                   self.task.render_action, self.task.action_title, self.task.render_reward,
                   (self.task.height, self.task.width), filename_video)
