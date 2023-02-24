import numpy as np
import copy
from single_agent_task import task_dim, BaseTask, PushTask, ThrowTask, CatchTask
from grid_policy import Policy
from grid_value import Critic


class PretrainedTask():
    def __init__(self, task_name, config):
        config['num_envs'] = 1
        config['num_agent'] = 1

        do_training = False
        if task_name == "push":
            self.task = PushTask(do_training, config)
        elif task_name == "throw":
            self.task = ThrowTask(do_training, config)
        elif task_name == "catch":
            self.task = CatchTask(do_training, config)
        arm_state_dim = task_dim[task_name]['arm_state']
        goal_dim = task_dim[task_name]['goal']
        task_action_dim = task_dim[task_name]['action']

        policy_config = {}
        policy_config['task'] = {'data_dir': "./pretrained_model/" + task_name + "/"}
        policy_config['env'] = {'num_envs': 1, 'num_agent': 1}
        policy_config['train'] = {'policy_lr': 0.0, 'critic_lr': 0.0}

        self.actor = Policy(task_action_dim, arm_state_dim, goal_dim,
                            load_actor=True, config=policy_config)

        # self.critic = Critic(arm_state_dim, goal_dim, load_critic=True, config=config)


class MultiAgentTask(BaseTask):
    def __init__(self, do_training, config):
        super().__init__(do_training, config)
        self.centralized_control = config['central']
        if self.centralized_control:
            self.goal_dim = 7  # each agent's action (1), ball / target info (6)
            total_goal_dim = self.goal_dim*self.n_agent+1 # all agents' goals, shooting status
            total_agent_state = self.n_agent*self.agent_state_dim
            self.goal_fn=self.centralized_goal_fn
            self.state_fn=self.centralized_state_fn
        else:
            self.goal_dim = 7 +2+ 2 * self.n_agent + 1 + self.n_agent*2  # ball / target info + shooting, agent pos, id, action, ball_in_agent
            total_goal_dim = self.goal_dim
            total_agent_state = self.agent_state_dim
            self.goal_fn=self.decentralized_goal_fn
        self.actor_goals = np.zeros((self.n_agent, self.goal_dim))
        self.action_dim = 3
        self.ball_in_agents = np.zeros(self.n_agent, dtype=bool)
        self.agent_rewards = np.zeros(self.n_agent)
        # self.low_lelve

        self.set_dimensions(self.action_dim, total_agent_state, total_goal_dim)
        self.episode_max_length = 600
        self.success_value = 1.0
        self.config = config

    def set_pretrained_task(self):
        self.throw = PretrainedTask("throw", self.config)
        self.throw.task.set_rng(self.rng)
        self.throw.task.task_reset()
        self.catch = PretrainedTask("catch", self.config)
        self.catch.task.set_rng(self.rng)
        self.catch.task.task_reset()
        self.push = PretrainedTask("push", self.config)
        self.push.task.set_rng(self.rng)
        self.push.task.task_reset()

    def task_reset(self):

        self.grid[...] = 0.0

        self.random_agent_start()
        self.pre_agent_location = copy.deepcopy(self.agent_states)
        self.episode_step = 0
        self.task_complete = False
        self.shooting_a_goal = False
        self.last_value = 0.0
        self.reward = 0.0
        self.action[...] = -1
        self.agent_rewards[...] = 0.0

        self.ball_in_target = False

        self.random_ball_start()
        self.random_target_start()
        self.random_ball_dir = 4

        # for i_agent in range(self.n_agent):
        #     agent_x, agent_y = self.agent_states[i_agent]
        #     all_ball_pos = [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x + 1, agent_y],
        #                     [agent_x, agent_y + 1]]
        # self.ball_in_agents[i_agent] = list(self.ball_location) in all_ball_pos

        ''' Reset ball location'''
        self.ball_location = self.ball_start[:2]
        self.pre_ball_location = copy.deepcopy(self.ball_location)

        ''' Reset target ball distance'''
        self.target_direction_ball = self.target_center - self.ball_location
        self.pre_target_direction_ball = copy.deepcopy(self.target_direction_ball)

        if self.render_log:
            self.update_render_log()

    def decentralized_goal_fn(self):
        self.episode_step += 1
        for i_agent in range(self.n_agent):
            agent_x, agent_y = self.agent_states[i_agent]
            all_ball_pos = [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x + 1, agent_y],
                            [agent_x, agent_y + 1]]
            self.ball_in_agents[i_agent] = list(self.ball_location) in all_ball_pos

        # TODO: add ball remain steps
        for i_agent, agent in enumerate(self.agent_states):
            ball_direction_agent = self.ball_location - agent
            pre_ball_direction_agent = self.pre_ball_location - self.pre_agent_location[i_agent]
            target_direction_agent = self.target_center - agent
            self.actor_goals[i_agent, ...] = np.concatenate([
                self.agent_states.flatten(),
                agent-self.pre_agent_location[i_agent],
                [i_agent],
                self.action,
                ball_direction_agent / self.width,
                pre_ball_direction_agent / self.width,
                target_direction_agent / self.width,
                [self.shooting_a_goal],
                self.ball_in_agents,
            ], axis=0)
        # print(self.actor_goals)
        # input("do it")

    def centralized_goal_fn(self):
        self.episode_step += 1
        # TODO: add ball remain steps
        for i_agent, agent in enumerate(self.agent_states):
            ball_direction_agent = self.ball_location - agent
            pre_ball_direciton_agent = self.pre_ball_location - agent
            target_direction_agent = self.target_center - agent
            self.actor_goals[i_agent, ...] = np.concatenate([
                self.action,
                ball_direction_agent / self.width,
                pre_ball_direciton_agent / self.width,
                target_direction_agent / self.width,
            ], axis=0)

    def centralized_state_fn(self):
        self.goal_fn()
        return np.hstack([self.agent_states.flatten(), self.actor_goals.flatten(), [self.shooting_a_goal]])

    def reward_fn(self):
        self.update_target_ball()
        self.ball_in_target = all(self.ball_location == self.target_center)
        self.reward = -0.1+self.shooting_a_goal+self.ball_in_target#(-0.1 + np.sum(self.agent_rewards)) * (1 - self.shooting_a_goal) + self.shooting_a_goal
        return self.reward

    def set_action_title(self):
        return ['throw', 'catch', 'push', 'start']

    def do_action(self, a, i_agent):
        self.pre_agent_location[i_agent]=self.agent_states[i_agent]
        if a == 0:  # throw
            policy_weight = 0.8
            self.do_action_with_pretrained_policy(self.throw, i_agent, policy_weight)
        elif a == 1:  # catch
            policy_weight = 0.01
            self.do_action_with_pretrained_policy(self.catch, i_agent, policy_weight)
        elif a == 2:  # push
            policy_weight = 0.19
            self.do_action_with_pretrained_policy(self.push, i_agent, policy_weight)
        else:
            raise NotImplementedError

    def do_action_with_pretrained_policy(self, pretrained_policy, i_agent, policy_weight):
        '''
        for each action
        (1) update parameters in pre-trained policy (low-level) using current global parameters (this class)
        (2) call state_fn in pre-trained policy to get the input for (3)
        (3) calculate low-level action using sample_action in pre-trained policy with (2)
        (4) apply this low-level action using do_action in task
        (5) update global parameters in high-level task (this class)
        '''
        # (1) update_pretrained_task_params
        pretrained_policy.task.grid[...] = self.grid
        pretrained_policy.task.agent_states[...] = self.agent_states[i_agent]
        # pretrained_policy.task.actor_goals[...] = self.actor_goals[i_agent]
        pretrained_policy.task.ball_location[...] = self.ball_location
        pretrained_policy.task.pre_ball_location[...] = self.pre_ball_location
        pretrained_policy.task.target_direction_ball[...] = self.target_direction_ball
        pretrained_policy.task.pre_target_direction_ball[...] = self.pre_target_direction_ball
        pretrained_policy.task.random_ball_dir = self.random_ball_dir
        pretrained_policy.task.target_center[...] = self.target_center
        pretrained_policy.task.check_ball_in_agent(self.ball_location)
        pretrained_policy.task.shooting_a_goal = self.shooting_a_goal
        pretrained_policy.task.ball_move_count = self.ball_move_count

        # (2) call state_fn
        observationNgoal = pretrained_policy.task.state_fn()

        # (3) calculate low-level action
        low_level_action, _ = pretrained_policy.actor.sample_action(observationNgoal)

        # (4) apply this low-level action using do_action in task
        pretrained_policy.task.do_action(int(low_level_action), 0)
        reward = pretrained_policy.task.task_reward()
        self.contribute = pretrained_policy.task.contribute
        # if self.shooting_a_goal and contribute:
        #     reward -= 1
        self.shooting_a_goal = pretrained_policy.task.shooting_a_goal if self.contribute else self.shooting_a_goal
        # print("->", contribute, pretrained_policy.task.ball_in_agent)
        # (5) update global parameters in high-level task (this class)
        self.grid[...] = pretrained_policy.task.grid
        self.agent_states[i_agent][...] = pretrained_policy.task.agent_states
        # self.ball_in_agents[i_agent] = pretrained_policy.task.ball_in_agent
        self.agent_rewards[i_agent] = reward * policy_weight
        # self.actor_goals[i_agent][...] = pretrained_policy.task.actor_goals
        self.ball_location[...] = pretrained_policy.task.ball_location
        # self.pre_ball_location[...] = pretrained_policy.task.pre_ball_location
        self.random_ball_dir = pretrained_policy.task.random_ball_dir
        self.ball_out_range = pretrained_policy.task.ball_out_range
        self.ball_in_target = pretrained_policy.task.ball_in_target

