import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from model import Actor
from tensorflow import keras


class Policy:
    def __init__(self, task_action_dim,
                 agent_state_dim, goal_dim,
                 load_actor, config):
        # dir
        train_dir = config['task']['data_dir']
        self.actor_dir = train_dir + '/actor_checkpoints/'

        # model
        self.num_task_actions = 1  # task_action_dim different defn in grid world
        self.num_task_categories = self.num_task_actions * task_action_dim
        self.task_action_dim = task_action_dim

        self.actor_model = Actor(self.num_task_categories)
        if load_actor:
            self.actor_model.load_weights(self.actor_dir + '/my_checkpoint')

        # parameters
        self.num_envs = config['env']['num_envs']
        self.num_agents = config['env']['num_agent']
        self.agent_state_dim = agent_state_dim
        self.goal_dim = goal_dim
        self.action_dim = 1  # task_action_dim

        self.clip_ratio = 0.2

        policy_learning_rate = config['train']['policy_lr']
        self.opt = keras.optimizers.Adam(learning_rate=policy_learning_rate)

    def sample_action(self, observationNgoal):
        observation = observationNgoal[..., :self.agent_state_dim]
        goal = observationNgoal[..., self.agent_state_dim:]
        observation = np.reshape(observation, (self.num_envs * self.num_agents, self.agent_state_dim))
        goal = np.reshape(goal, (self.num_envs * self.num_agents, self.goal_dim))
        logits = self.actor_model(observation, goal)  # (batch*arm,4*task)
        logits = tf.reshape(logits, (-1, self.task_action_dim))  # (batch*arm*task,4)
        task_action = tf.random.categorical(logits, 1)  # (batch*arm*task,1)
        task_log_pi = self.logprobabilities(logits, task_action)  # (batch*arm*task,1)

        task_action = np.reshape(task_action, (self.num_envs, self.num_agents, self.num_task_actions))

        task_log_pi = np.reshape(task_log_pi,
                                 (self.num_envs, self.num_agents, self.num_task_actions))  # (batch, arm,task)

        return task_action, task_log_pi

    # @tf.function
    def train_policy(
            self, samples
    ):
        batch_obs, batch_goal, batch_act, batch_log, batch_adv = samples
        batch_task_act = tf.cast(batch_act, tf.int32)
        batch_task_act = tf.reshape(batch_task_act, (-1, 1))  # (batch*arm*task,1)
        batch_task_log = tf.reshape(batch_log, (-1,))

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            tape.watch(self.actor_model.trainable_variables)
            new_logits = self.actor_model(batch_obs, batch_goal)[:, -self.num_task_categories:]  # (batch*arm,4*task)
            new_logits = tf.reshape(new_logits, (-1, self.task_action_dim))  # (batch*arm*task,4)
            task_log_pi = self.logprobabilities(new_logits, batch_task_act)[:,
                          0]  # (batch*arm*task,1)->(batch*arm*task)
            task_policy_loss = self.cal_policy_loss(task_log_pi, batch_task_log, batch_adv)

        policy_grads = tape.gradient(task_policy_loss,
                                     sources=self.actor_model.trainable_variables)
        self.opt.apply_gradients(
            zip(policy_grads, self.actor_model.trainable_variables))

        kl = tf.reduce_mean(batch_task_log - task_log_pi)

        return kl, task_policy_loss

    @tf.function
    def cal_policy_loss(self, log_pi, batch_log, batch_adv):
        ratio = tf.exp(log_pi - batch_log)

        flatten_min_adv = tf.where(
            batch_adv > 0,
            (1 + self.clip_ratio) * batch_adv,
            (1 - self.clip_ratio) * batch_adv,
        )
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * batch_adv, flatten_min_adv)
        )
        return policy_loss

    def save(self):
        self.actor_model.save_weights(self.actor_dir + '/my_checkpoint')

    # @tf.function
    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)[:, tf.newaxis, :]  # new axis for one hot
        # one hot action: tf.one_hot(a,4) -> shape (batch,1,4)   control 4 dirs
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.task_action_dim) * logprobabilities_all, axis=-1
        )  # shape (batch,1)
        return logprobability


class CentralPolicy:
    def __init__(self, task_action_dim,
                 agent_state_dim, goal_dim,
                 load_actor, config):
        # dir
        train_dir = config['task']['data_dir']
        self.actor_dir = train_dir + '/actor_checkpoints/'

        # parameters
        self.num_envs = config['env']['num_envs']
        self.num_agents = config['env']['num_agent']
        self.agent_state_dim = agent_state_dim
        self.goal_dim = goal_dim
        self.action_dim = 1  # task_action_dim

        # model
        self.num_task_actions = 1  # task_action_dim different defn in grid world
        self.num_task_categories = task_action_dim**self.num_agents
        self.task_action_dim = self.num_task_categories

        self.actor_model = Actor(self.num_task_categories)
        if load_actor:
            self.actor_model.load_weights(self.actor_dir + '/my_checkpoint')

        self.clip_ratio = 0.2
        policy_learning_rate = config['train']['policy_lr']
        self.opt = keras.optimizers.Adam(learning_rate=policy_learning_rate)

    def sample_action(self, observationNgoal):
        observation = observationNgoal[..., :self.agent_state_dim].reshape((self.num_envs,2*self.num_agents))
        goal = observationNgoal[..., self.agent_state_dim:]
        act = goal[:,0,2 * self.num_agents + 1:2 * self.num_agents + 1 + self.num_agents] # extract actions from the first agent info
        act = np.reshape(act, (self.num_envs,self.num_agents))
        all_agent_goals = goal[..., -7:].reshape((self.num_envs,7*self.num_agents))
        goals = np.hstack([act,all_agent_goals])
        logits = self.actor_model(observation,goals)
        task_action = tf.random.categorical(logits, 1)  # (batch*arm*task,1)
        task_log_pi = self.logprobabilities(logits, task_action)
        ternary=[]
        for a in task_action:
            t=np.base_repr(a[0], base=3)
            t='0' * (self.num_agents - len(t)) + t
            ternary.append([int(d) for d in t])
        # ternary = [np.base_repr(a[0], base=3) for a in task_action]
        # ternary = ['0' * (self.num_agents - len(ternary)) + ternary]
        # task_action = [int(a) for a in ternary]

        task_action = np.reshape(ternary, (self.num_envs, self.num_agents, self.num_task_actions))

        task_log_pi = np.reshape(task_log_pi,
                                 (self.num_envs, 1, self.num_task_actions))  # (batch, arm,task)

        return task_action, task_log_pi

    # @tf.function
    def train_policy(
            self, samples
    ):
        batch_obs, batch_goal, batch_act, batch_log, batch_adv = samples
        batch_task_act = tf.cast(batch_act, tf.int32)
        batch_task_act = tf.reshape(batch_task_act, (-1, 1))  # (batch*arm*task,1)
        batch_task_log = tf.reshape(batch_log, (-1,))

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            tape.watch(self.actor_model.trainable_variables)
            new_logits = self.actor_model(batch_obs, batch_goal)[:, -self.num_task_categories:]  # (batch*arm,4*task)
            new_logits = tf.reshape(new_logits, (-1, self.task_action_dim))  # (batch*arm*task,4)
            task_log_pi = self.logprobabilities(new_logits, batch_task_act)[:,
                          0]  # (batch*arm*task,1)->(batch*arm*task)
            task_policy_loss = self.cal_policy_loss(task_log_pi, batch_task_log, batch_adv)

        policy_grads = tape.gradient(task_policy_loss,
                                     sources=self.actor_model.trainable_variables)
        self.opt.apply_gradients(
            zip(policy_grads, self.actor_model.trainable_variables))

        kl = tf.reduce_mean(batch_task_log - task_log_pi)

        return kl, task_policy_loss

    @tf.function
    def cal_policy_loss(self, log_pi, batch_log, batch_adv):
        ratio = tf.exp(log_pi - batch_log)

        flatten_min_adv = tf.where(
            batch_adv > 0,
            (1 + self.clip_ratio) * batch_adv,
            (1 - self.clip_ratio) * batch_adv,
        )
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * batch_adv, flatten_min_adv)
        )
        return policy_loss

    def save(self):
        self.actor_model.save_weights(self.actor_dir + '/my_checkpoint')

    # @tf.function
    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)[:, tf.newaxis, :]  # new axis for one hot
        # one hot action: tf.one_hot(a,4) -> shape (batch,1,4)   control 4 dirs
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.task_action_dim) * logprobabilities_all, axis=-1
        )  # shape (batch,1)
        return logprobability