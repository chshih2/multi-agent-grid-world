import numpy as np
import scipy
import scipy.signal
import tensorflow as tf


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, size, num_actions, num_task_actions,
                 arm_state_dim, goal_dim,
                 num_envs=1, num_arms=1):
        gamma = 0.99
        lam = 0.95
        self.num_envs = num_envs
        # Buffer initialization
        self.num_arms = num_arms
        self.arm_state_dim = arm_state_dim
        self.goal_dim = goal_dim
        self.action_dim = num_actions
        self.task_action_dim = num_task_actions
        self.obsNgoal_dim = self.arm_state_dim + self.goal_dim
        self.observationNgoal_buffer = np.zeros(
            (num_envs, size, num_arms, self.obsNgoal_dim), dtype=np.float32
        )
        self.action_buffer = np.zeros((num_envs, size, num_arms, num_actions + num_task_actions), dtype=np.float32)
        # self.task_action_buffer = np.zeros((num_envs, size, num_arms, num_task_actions), dtype=np.int32)
        self.advantage_buffer = np.zeros((num_envs, size), dtype=np.float32)
        self.reward_buffer = np.zeros((num_envs, size), dtype=np.float32)
        self.return_buffer = np.zeros((num_envs, size), dtype=np.float32)
        self.value_buffer = np.zeros((num_envs, size), dtype=np.float32)
        self.logprobability_buffer = np.zeros((num_envs, size, num_arms, num_task_actions), dtype=np.float32)
        # self.task_logprobability_buffer = np.zeros((num_envs, size, num_arms), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer = np.zeros((num_envs,), dtype=int)
        self.trajectory_start_index = np.zeros((num_envs,), dtype=int)
        self.num_envs = num_envs

    def store(self, observationNgoal, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        for env_idx in range(self.num_envs):

            self.observationNgoal_buffer[env_idx, self.pointer[env_idx]] = observationNgoal[env_idx]
            self.action_buffer[env_idx, self.pointer[env_idx]] = action[env_idx]
            self.reward_buffer[env_idx, self.pointer[env_idx]] = reward[env_idx]
            self.value_buffer[env_idx, self.pointer[env_idx]] = value[env_idx]
            # TODO: fix logprobability shape (along with policy sample_action)
            self.logprobability_buffer[env_idx, self.pointer[env_idx]] = logprobability[env_idx] # centralized policy will store duplicated logprob
        self.pointer += 1

    def finish_trajectory(self, env_idx, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index[env_idx], self.pointer[env_idx])
        rewards = np.append(self.reward_buffer[env_idx, path_slice], last_value)
        values = np.append(self.value_buffer[env_idx, path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[env_idx, path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[env_idx, path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index[env_idx] = self.pointer[env_idx]

    def cal_advantage_buffer(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer = np.zeros((self.num_envs,), dtype=int)
        self.trajectory_start_index = np.zeros((self.num_envs,), dtype=int)
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer, axis=-1)[:, np.newaxis],
            np.std(self.advantage_buffer, axis=-1)[:, np.newaxis],
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std

    def stack_up(self):
        self.vstack_obs = np.reshape(self.observationNgoal_buffer[:, :, :, :self.arm_state_dim],
                                     [-1, self.arm_state_dim])
        self.vstack_goal = np.reshape(self.observationNgoal_buffer[:, :, :, self.arm_state_dim:], [-1, self.goal_dim])
        self.vstack_action = np.reshape(self.action_buffer, [-1, self.task_action_dim])

        self.flatten_logp = np.reshape(self.logprobability_buffer, [-1, self.task_action_dim])
        self.flatten_adv = np.reshape(self.advantage_buffer, [-1])

    def sample(self, sample_idx):
        batch_obs = tf.gather(self.vstack_obs, sample_idx)
        batch_goal = tf.gather(self.vstack_goal, sample_idx)
        batch_act = tf.gather(self.vstack_action, sample_idx)
        batch_log = tf.gather(self.flatten_logp, sample_idx)
        batch_adv = tf.gather(self.flatten_adv, sample_idx)
        return batch_obs, batch_goal, batch_act, batch_log, batch_adv
