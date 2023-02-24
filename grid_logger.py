import numpy as np
from tensorboardX import SummaryWriter


class LogProgress:
    def __init__(self, num_envs, train_dir, test_log):
        self.train_writer = SummaryWriter(train_dir)
        self.sum_return = np.zeros(num_envs)
        # self.sum_u = np.zeros(num_envs)
        self.sum_length = np.zeros(num_envs, dtype=int)
        self.sum_success = np.zeros(num_envs, dtype=int)
        self.num_episodes = np.zeros(num_envs, dtype=int)

        if test_log:
            self.weights = []
            self.primitives = []
            self.activations = []

    def reset(self):
        self.sum_return[...] = 0.0
        # self.sum_u[...] = 0.0
        self.sum_length[...] = 0
        self.sum_success[...] = 0
        self.num_episodes[...] = 0

    def count(self, reward):
        self.sum_return += reward
        # self.sum_u += control_cost
        self.sum_length += 1

    def reset_decomposition(self):
        self.weights = []
        self.primitives = []
        self.activations = []

    def log_decomposition(self, weight, primitive, activation):
        self.weights.append(weight)
        self.primitives.append(primitive)
        self.activations.append(activation)

    def end_episode(self, env_idx, success):
        self.sum_success[env_idx] += int(success)
        self.num_episodes[env_idx] += 1

    def end_epoch(self, epoch, value_loss, policy_loss, kl):
        mean_return=np.mean(self.sum_return / self.num_episodes)
        mean_success=np.mean(self.sum_success / self.num_episodes)
        print(
            f"Epoch: {epoch + 1}. Mean Return: {mean_return}. Mean Success: {mean_success}."
            f"Value Loss: {value_loss}. Policy Loss: {policy_loss}. KL: {kl}"
        )
        assert not np.isnan(policy_loss), "policy Nan"
        assert not np.isnan(value_loss), "value Nan"

        self.train_writer.add_scalar("mean-return", mean_return, epoch)
        # self.train_writer.add_scalar("mean-u (per steps)", np.mean(self.sum_u / self.num_episodes / self.sum_length),
        #                              epoch)
        self.train_writer.add_scalar("mean-success", mean_success, epoch)
        self.train_writer.add_scalar("mean-eps-length", np.mean(self.sum_length / self.num_episodes), epoch)
        self.train_writer.add_scalar("kl", kl, epoch)
        self.train_writer.add_scalar("policy_loss", policy_loss, epoch)
        self.train_writer.add_scalar("value_loss", value_loss, epoch)

    def end_test(self, filename):
        print(
            f"{filename}. Mean Return: {self.sum_return / self.num_episodes}."
        )
        result = {"return": np.mean(self.sum_return / self.num_episodes),
                  # "u": np.mean(self.sum_u / self.num_episodes / self.sum_length),
                  "success": np.mean(self.sum_success / self.num_episodes),
                  "eps-length": np.mean(self.sum_length / self.num_episodes)}
        np.savez(filename + "_log", **result)
