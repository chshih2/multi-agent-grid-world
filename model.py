from tensorflow import keras
import tensorflow as tf


class Actor(keras.Model):
    def __init__(self, num_task_actions):
        super().__init__()
        self.s1 = keras.layers.Dense(256, activation=tf.nn.relu)
        self.s2 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.g1 = keras.layers.Dense(256, activation=tf.nn.relu)
        self.g2 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.sg1 = keras.layers.Concatenate()
        self.sg2 = keras.layers.Dense(128, activation=tf.nn.relu)

        self.num_task_actions = num_task_actions
        self.o2 = keras.layers.Dense(num_task_actions)

    # @tf.function
    def call(self, state, goal):
        s = self.s1(state)
        s = self.s2(s)
        g = self.g1(goal)
        g = self.g2(g)
        sg = self.sg1([s, g])
        sg = self.sg2(sg)
        return self.o2(sg)


class Critic_mlp(keras.Model):
    def __init__(self):
        super().__init__()
        self.s1 = keras.layers.Dense(256, activation=tf.nn.relu)
        self.s2 = keras.layers.Dense(64, activation=tf.nn.relu)
        self.o = keras.layers.Dense(1)

    # @tf.function
    def call(self, state):
        s = self.s1(state)
        s = self.s2(s)
        return tf.squeeze(self.o(s), axis=1)
