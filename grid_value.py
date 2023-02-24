import tensorflow as tf
from tensorflow import keras
from model import Critic_mlp
import numpy as np


class Critic:
    def __init__(self, arm_state_dim, goal_dim, load_critic, config):
        self.obsNgoal_dim = arm_state_dim + goal_dim
        self.obsNgoal_shape = [-1, self.obsNgoal_dim]
        self.num_envs = config['env']['num_envs']
        self.num_arms = config['env']['num_agent']

        self.critic = Critic_mlp()
        train_dir = config['task']['data_dir']
        self.critic_dir = train_dir + '/critic_checkpoints/'

        if load_critic:
            self.critic.load_weights(self.critic_dir + '/my_checkpoint')

        value_learning_rate = config['train']['critic_lr']
        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_learning_rate)

    # @tf.function
    # def __call__(self, obsNgoal):
    #     value_t = self.critic(obsNgoal)
    #     return value_t
    def get_value(self, obsNgoal):
        obsNgoal = np.reshape(obsNgoal, (self.num_envs*self.num_arms, self.obsNgoal_dim))
        value_t = self.critic(obsNgoal)

        '''naive multi-agent'''
        value_t = np.reshape(value_t, (self.num_envs, self.num_arms)).mean(axis=-1)
        return value_t

    # Train the value function by regression on mean-squared error
    def train_value_function(self, observationNgoal_buffer, return_buffer):
        vstack_obsNgoal = np.reshape(observationNgoal_buffer,[-1, self.obsNgoal_dim])
        vstack_return = np.reshape(return_buffer, [-1])
        vstack_return = np.repeat(vstack_return, self.num_arms)
        value_loss = self.apply_value_loss(vstack_return,vstack_obsNgoal)
        return value_loss
    @tf.function
    def apply_value_loss(self,vstack_return, vstack_obsNgoal):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((vstack_return - self.critic(vstack_obsNgoal)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))
        return value_loss

    def save(self):
        self.critic.save_weights(self.critic_dir + '/my_checkpoint')
