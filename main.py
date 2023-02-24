import numpy as np
import tensorflow as tf
import shutil
import copy
import yaml
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import gym
from tqdm import tqdm
from set_vec_environment import Environment
from task_config import get_task
from grid_policy import Policy, CentralPolicy
from grid_value import Critic
from grid_buffer import Buffer
from grid_logger import LogProgress


def build_pipeline(config):
    do_training = config['task']['do_train']
    num_envs = config['env']['num_envs']
    centralized_policy=config['env']['central']

    ''' Create envs '''
    task = get_task(do_training, config['env'])
    if do_training:
        def env_lambda(seed):
            return lambda: Environment(task, env_config=config['env'], seed=seed)

        env_fn = [env_lambda(i) for i in range(num_envs)]
        envs = gym.vector.AsyncVectorEnv(env_fn)
    else:
        envs = Environment(task, env_config=config['env'], seed=100)

    ''' Initialize the buffer '''
    arm_state_dim, goal_dim = task.agent_state_dim, task.goal_dim
    task_action_dim = task.action_dim
    steps_per_epoch = config['train']['steps_per_epoch']
    num_arms = config['env']['num_agent']
    buffer = Buffer(size=steps_per_epoch, num_actions=0, num_task_actions=1,
                    arm_state_dim=arm_state_dim, goal_dim=goal_dim,
                    num_envs=num_envs, num_arms=num_arms)

    # set load flag
    load_actor, load_critic = get_load_flag(do_training)

    # Initialize the policy
    if centralized_policy:
        actor = CentralPolicy(task_action_dim, arm_state_dim, goal_dim,
                              load_actor, config)
    else:
        actor = Policy(task_action_dim, arm_state_dim, goal_dim,
                       load_actor, config)

    # Initialize the value function
    critic = Critic(arm_state_dim, goal_dim, load_critic, config)
    return envs, buffer, actor, critic


def train(env, buffer, agent, critic, config):
    train_dir = config['task']['data_dir']
    num_envs = config['env']['num_envs']
    epochs = config['train']['epochs']
    steps_per_epoch = config['train']['steps_per_epoch']
    train_policy_iterations = 80
    train_value_iterations = 80
    rollout_size = steps_per_epoch * num_envs
    batch_size = 4000
    target_kl = 0.01

    observationNgoal = env.reset()

    logger = LogProgress(num_envs, train_dir, test_log=False)

    for epoch in range(epochs):
        logger.reset()

        # Iterate over the steps of each epoch
        for t in tqdm(range(steps_per_epoch)):
            action, logprobability_t = agent.sample_action(observationNgoal)
            observationNgoal_new, reward, done, info = env.step(action)
            value_t = critic.get_value(observationNgoal)
            buffer.store(observationNgoal, action, reward, value_t, logprobability_t)
            observationNgoal = observationNgoal_new

            logger.count(reward)

            end_of_epoch = t == steps_per_epoch - 1
            if end_of_epoch:
                last_values = critic.get_value(observationNgoal)
                for env_idx, last_value in enumerate(last_values):
                    buffer.finish_trajectory(env_idx, last_value)
                    logger.end_episode(env_idx, info[env_idx]['success'])
            elif done.any():
                for env_idx in range(num_envs):
                    if done[env_idx]:
                        last_value = info[env_idx]['last_value']
                        buffer.finish_trajectory(env_idx, last_value)
                        logger.end_episode(env_idx, info[env_idx]['success'])

        # Get values from the buffer
        buffer.cal_advantage_buffer()

        # Update the policy and implement early stopping using KL divergence
        buffer.stack_up()
        for _ in range(train_policy_iterations):
            sample_idx = np.random.randint(rollout_size, size=batch_size)
            samples = buffer.sample(sample_idx)
            kl, policy_loss = agent.train_policy(samples)
            if kl > 1.5 * target_kl:
                break

        # Update the value function
        for _ in range(train_value_iterations):
            value_loss = critic.train_value_function(buffer.observationNgoal_buffer, buffer.return_buffer)

        logger.end_epoch(epoch, value_loss.numpy(), policy_loss.numpy(), kl.numpy())

        agent.save()
        critic.save()


def test(env, agent, config):
    save_dir = config['task']['data_dir']
    num_envs = config['env']['num_envs']
    env.task.test_setup()

    logger = LogProgress(num_envs, train_dir, test_log=True)

    for test_idx in range(env.task.num_tests):
        env.task.test_next()
        observationNgoal = env.reset()
        logger.reset()
        # logger.reset_decomposition()
        for _ in range(env.task.episode_max_length):
            action, logprobability_t = agent.sample_action(observationNgoal)
            observationNgoal_new, reward, done, info = env.step(action[0])
            observationNgoal = observationNgoal_new
            logger.count(reward)
            # logger.log_decomposition(weights, primitives, copy.deepcopy(env.arm_simulator.u))
            if done:
                logger.end_episode(env_idx=0, success=info['success'])
                break
        test_name = save_dir + "/test%d" % test_idx
        logger.end_test(filename=test_name)
        env.save_data(filename=test_name)


def get_load_flag(do_training):
    load_actor, load_critic = True, True
    if do_training:
        load_actor = False
        load_critic = False
    return load_actor, load_critic


if __name__ == '__main__':

    # read config from yaml file
    yaml_dir = './my_config.yaml'
    with open(yaml_dir) as file:
        run_config = yaml.safe_load(file)

    env, buffer, agent, critic = build_pipeline(run_config)
    do_training = run_config['task']['do_train']
    train_dir = run_config['task']['data_dir']

    if do_training:
        os.makedirs(train_dir, exist_ok=True)
        shutil.copyfile(yaml_dir, train_dir + "/config.yaml")
        train(env, buffer, agent, critic, run_config)
    else:
        test(env, agent, run_config)
