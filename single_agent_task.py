import numpy as np
import gym
import copy

task_dim = {
    'push': {'arm_state': 2, 'goal': 9, 'action': 4},
    'throw': {'arm_state': 2, 'goal': 9, 'action': 5},
    'catch': {'arm_state': 2, 'goal': 9, 'action': 5},
}


class BaseTask():
    '''
    In the world (grid),
    1: agent
    2: ball
    3: target
    '''

    def __init__(self, do_training, config):
        self.do_training = do_training
        self.config = config

        self.n_agent = config['num_agent']
        self.action_dim = 1
        self.agent_state_dim = 2
        self.agent_states = np.zeros((self.n_agent, 2), dtype=int)

        ''' Ball / target task '''
        self.ball_start = None
        self.ball_start_vel = None
        self.ball_in_target = False
        self.random_ball_dir = 4  # still
        self.ball_location = np.zeros(2, dtype=int)
        self.pre_ball_location = np.zeros(2, dtype=int)
        self.target_center = np.zeros(2, dtype=int)
        self.ball_in_agent = False
        self.shooting_a_goal = False

        ''' World '''
        self.width = 10
        self.height = 8
        self.grid = np.zeros((self.height, self.width), dtype=int)

        ''' General task '''
        self.task_complete = False
        self.episode_max_length = 0
        self.episode_step = 0
        self.render_log = False

    def set_target_box(self, box_start, box_dim):
        self.target_box_start = box_start  # [0.05, -0.1]
        self.target_box_dim = box_dim  # [0.15, 0.25]

    def set_dimensions(self, action_dim, agent_state_dim, goal_dim):
        # (arm) action
        # action_size = (action_dim,)
        # action_low = np.ones(action_size) * (-1)
        # action_high = np.ones(action_size) * (1)
        self.action_space = gym.spaces.Discrete(action_dim)
        # self.action_space = gym.spaces.Box(action_low, action_high, shape=action_size, dtype=np.float32)
        self.action = np.ones(self.n_agent, dtype=int) * (-1)
        # state
        observation_size = (self.n_agent, agent_state_dim + goal_dim)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=observation_size, dtype=np.float32)

    def state_fn(self):
        self.goal_fn()
        return np.hstack([self.agent_states, self.actor_goals])

    def teset_setup(self):
        raise NotImplementedError

    def task_reset(self):
        raise NotImplementedError

    def task_done(self):
        if self.render_log:
            self.update_render_log()
        # ball_out_range
        if self.ball_out_range:
            self.last_value = 0.0
            return True
        # ball_in_target
        elif self.ball_in_target:
            self.last_value = self.success_value
            self.task_complete = True
            return True
        # end_of_eps
        elif self.episode_step == self.episode_max_length:
            self.last_value = 0.0
            return True
        return False

    def reward_fn(self):
        reward = self.task_reward()
        self.reward = reward#-0.1 + reward - self.ball_out_range
        return self.reward

    def action_fn(self, action):

        self.action[...] = action.flatten()
        self.pre_ball_location[...] = self.ball_location
        self.ball_move_count = 0
        no_touch_count = 0
        for i_agent, a in enumerate(action):
            self.do_action(a, i_agent)

            # if self.ball_move_count == 0 and not self.ball_in_target and not self.contribute and self.random_ball_dir != 4:
            #     self.free_ball_dynamics()
            #     self.ball_move_count += 1
            # elif self.contribute:
            #     self.ball_move_count += 1
            self.ball_in_target = all(self.ball_location == self.target_center)
            if self.ball_in_target:
                return
            if not self.contribute:
                no_touch_count += 1
        if no_touch_count == self.n_agent:
            self.free_ball_dynamics()
            self.ball_out_range = self.ball_location[0] == 0 or self.ball_location[0] == self.height - 1 or \
                                  self.ball_location[1] == 0 or self.ball_location[1] == self.width - 1
            self.ball_in_target = all(self.ball_location == self.target_center)
        self.ball_in_agent = self.check_ball_in_agent(self.ball_location)
        # ball_x, ball_y = self.ball_location
        # self.grid[self.ball_location[0], self.ball_location[1]] = 0
        # next_ball_location = self.cal_next_ball_loc(self.ball_location)
        # self.ball_in_target = all(next_ball_location == self.target_center)
        # if self.ball_in_target:
        #     self.ball_location[...] = next_ball_location
        # else:
        #     self.ball_location = self.cal_next_ball_loc(next_ball_location)
        # self.grid[self.ball_location[0], self.ball_location[1]] = 2
        # self.check_ball_in_agent()
        #
        # self.grid[ball_x, ball_y] = 0
        # self.grid[self.ball_location[0], self.ball_location[1]] = 2
        # self.ball_out_range = self.ball_location[0] == 0 or self.ball_location[0] == self.height - 1 or \
        #                       self.ball_location[1] == 0 or self.ball_location[1] == self.width - 1

    def goal_fn(self):
        self.episode_step += 1
        # TODO: add ball remain steps
        for i_agent, agent in enumerate(self.agent_states):
            ball_direction_agent = self.ball_location - agent
            pre_ball_direciton_agent = self.pre_ball_location - self.pre_agent_location
            target_direction_agent = self.target_center - agent
            self.actor_goals[i_agent, ...] = np.concatenate([
                ball_direction_agent / self.width,
                pre_ball_direciton_agent / self.width,
                target_direction_agent / self.width,
                self.pre_agent_location,
                [self.shooting_a_goal]
            ], axis=0)
        # print(self.actor_goals)
        # input("do it")

    def set_rng(self, rng):
        self.rng = rng

    def random_agent_start(self):
        for i_agent in range(self.n_agent):
            occupied = True
            while occupied:
                agent_x = self.rng.integers(1, self.height - 1)
                agent_y = self.rng.integers(1, self.width - 1)
                occupied = self.grid[agent_x][agent_y] != 0
            # agent_x,agent_y=1,3
            self.grid[agent_x][agent_y] = 1
            self.agent_states[i_agent, ...] = [agent_x, agent_y]

    def random_ball_start(self):
        occupied = True
        while occupied:
            ball_x = self.rng.integers(1, self.height - 1)
            ball_y = self.rng.integers(1, self.width - 1)
            occupied = self.grid[ball_x][ball_y] != 0
        # ball_x,ball_y=2,1
        self.grid[ball_x][ball_y] = 2
        self.ball_start = np.array([ball_x, ball_y])

    def random_ball_start_far_from_agent(self):
        occupied = True
        agent_x, agent_y = self.agent_states[0]
        all_ball_pos = [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x + 1, agent_y], [agent_x, agent_y + 1]]
        while occupied:
            ball_x = self.rng.integers(1, self.height - 1)
            ball_y = self.rng.integers(1, self.width - 1)
            occupied = self.grid[ball_x][ball_y] != 0 or [ball_x, ball_y] in all_ball_pos
        self.grid[ball_x][ball_y] = 2
        self.ball_start = np.array([ball_x, ball_y])

    def random_ball_start_next_to_agent(self):
        agent_x, agent_y = self.agent_states[0]
        all_ball_pos = [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x + 1, agent_y], [agent_x, agent_y + 1]]
        available_ball_pos = []
        for ball_pos in all_ball_pos:
            if ball_pos[0] > 0 and ball_pos[0] < self.height - 1 and ball_pos[1] > 0 and ball_pos[1] < self.width - 1:
                available_ball_pos.append(ball_pos)
        ball_x, ball_y = available_ball_pos[self.rng.integers(len(available_ball_pos))]
        self.grid[ball_x][ball_y] = 2
        self.ball_start = np.array([ball_x, ball_y])

    def random_ball_start_align_with_agent(self):
        agent_x, agent_y = self.agent_states[0]
        all_ball_pos = [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x + 1, agent_y], [agent_x, agent_y + 1]]
        available_ball_pos = []
        for ball_pos in all_ball_pos:
            if ball_pos[0] > 0 and ball_pos[0] < self.height - 1 and ball_pos[1] > 0 and ball_pos[1] < self.width - 1:
                available_ball_pos.append(ball_pos)

        occupied = True
        count = 0
        while occupied and count < 15:
            count += 1
            x_or_y = self.rng.integers(2)
            ball_x, ball_y = available_ball_pos[self.rng.integers(len(available_ball_pos))]
            if x_or_y == 0:
                ball_y = self.rng.integers(1, self.width - 1)
                self.random_ball_dir = 1 if ball_y - self.agent_states[0][
                    1] >= 0 else 3  # 'down', 'left', 'up', 'right','still'

            else:
                ball_x = self.rng.integers(1, self.height - 1)
                self.random_ball_dir = 2 if ball_x - self.agent_states[0][0] >= 0 else 0

            occupied = self.grid[ball_x][ball_y] != 0
        self.grid[ball_x][ball_y] = 2
        self.ball_start = np.array([ball_x, ball_y])

    def random_target_start_align_ball(self):
        ball_x, ball_y = self.ball_start[:2]
        agent_x, agent_y = self.agent_states[0]

        occupied = True
        count = 0
        while occupied and count < 15:
            count += 1
            align_ball_or_agent = self.rng.integers(2)
            if self.random_ball_dir in [1, 3]:  # ball is moving left or right
                if align_ball_or_agent == 0:  # align ball
                    target_x = ball_x
                    # target_y = self.rng.integers(1, self.width - 1)
                    if ball_y >= agent_y:
                        target_y = self.rng.integers(ball_y + 1, self.width - 1) if ball_y < self.width - 2 else ball_y
                    else:
                        target_y = self.rng.integers(1, ball_y) if ball_y > 1 else ball_y
                else:
                    # target_x = self.rng.integers(1, self.height - 1)
                    if ball_x == agent_x:
                        target_y = agent_y - 1 if ball_y - agent_y < 0 else agent_y + 1
                    else:
                        target_y = agent_y
                    if ball_x <= agent_x:
                        target_x = self.rng.integers(1, ball_x) if ball_x > 1 else ball_x
                    else:
                        target_x = self.rng.integers(ball_x + 1,
                                                     self.height - 1) if ball_x < self.height - 2 else ball_x
            else:  # ball is moving up or down
                if align_ball_or_agent == 0:  # align ball
                    if ball_x >= agent_x:
                        target_x = self.rng.integers(ball_x + 1,
                                                     self.height - 1) if ball_x < self.height - 2 else ball_x
                    else:
                        target_x = self.rng.integers(1, ball_x) if ball_x > 1 else ball_x
                    target_y = ball_y
                else:
                    if ball_y == agent_y:
                        target_x = agent_x - 1 if ball_x - agent_x < 0 else agent_x + 1
                    else:
                        target_x = agent_x

                    if ball_y <= agent_y:
                        target_y = self.rng.integers(1, ball_y) if ball_y > 1 else ball_y
                    else:
                        target_y = self.rng.integers(ball_y + 1, self.width - 1) if ball_y < self.width - 2 else ball_y
            occupied = self.grid[target_x][target_y] != 0
        self.grid[target_x][target_y] = 3
        self.target_center = np.array([target_x, target_y])

    def random_target_start(self):
        occupied = True
        while occupied:
            target_x = self.rng.integers(1, self.height - 1)
            target_y = self.rng.integers(1, self.width - 1)
            occupied = self.grid[target_x][target_y] != 0
        # target_x,target_y=1,2
        self.grid[target_x][target_y] = 3
        self.target_center = np.array([target_x, target_y])

    def free_ball_dynamics(self):
        '''when the agent is not touching the ball'''
        self.grid[self.ball_location[0], self.ball_location[1]] = 0
        next_ball_location = self.cal_next_ball_loc(self.ball_location)
        self.ball_in_target = all(next_ball_location == self.target_center)
        if self.ball_in_target:
            self.ball_location[...] = next_ball_location
        else:
            self.ball_location = self.cal_next_ball_loc(next_ball_location)
        self.grid[self.ball_location[0], self.ball_location[1]] = 2
        self.ball_in_agent = self.check_ball_in_agent(self.ball_location)

    def check_align_ball_target(self):
        if self.ball_in_target:
            self.align_ball_target = True
            self.shooting_a_goal = self.contribute
        else:
            x_align = self.ball_location[0] == self.target_center[0]
            y_align = self.ball_location[1] == self.target_center[1]
            self.align_ball_target = False
            if (x_align or y_align):
                l1_dist_ball_target = np.linalg.norm(self.target_direction_ball, 1)
                l1_dist_agent_target = np.linalg.norm(self.target_center - self.agent_states[0], 1)
                if l1_dist_ball_target < l1_dist_agent_target:
                    self.align_ball_target = True
                    if self.contribute:
                        next_ball_location = self.cal_next_ball_loc(self.ball_location)
                        if all(next_ball_location == self.target_center):
                            self.shooting_a_goal = True
                        else:
                            next_x_align = x_align and next_ball_location[0] == self.target_center[0]
                            next_y_align = y_align and next_ball_location[1] == self.target_center[1]
                            next_l1_dist_ball_target = np.linalg.norm(self.target_center - next_ball_location, 1)
                            self.shooting_a_goal = next_l1_dist_ball_target < l1_dist_ball_target and (
                                    next_x_align or next_y_align)
                    else:
                        self.shooting_a_goal = self.shooting_a_goal

    def update_target_ball(self):
        # if self.render_log:
        #     self.update_render_log()
        ''' Update target ball distance and vel'''
        self.target_direction_ball = self.target_center - self.ball_location
        self.ball_dist_imprv = np.linalg.norm(self.pre_target_direction_ball, 1) - np.linalg.norm(
            self.target_direction_ball, 1)
        self.pre_target_direction_ball[...] = self.target_direction_ball

    def cal_pre_ball_loc(self):
        if self.random_ball_dir == 4:
            self.pre_ball_location = copy.deepcopy(self.ball_location)
        elif self.random_ball_dir == 0:  # ball is going down -> pre loc is up
            self.pre_ball_location = self.ball_location - np.array([2, 0])
        elif self.random_ball_dir == 1:  # ball is going left -> pre loc is right
            self.pre_ball_location = self.ball_location + np.array([0, 2])
        elif self.random_ball_dir == 2:  # ball is going up -> pre loc is down
            self.pre_ball_location = self.ball_location + np.array([2, 0])
        elif self.random_ball_dir == 3:  # ball is going right -> pre loc is left
            self.pre_ball_location = self.ball_location - np.array([0, 2])

    def cal_next_ball_loc(self, ball_location):
        ''' when ball is not moved by agent'''
        ball_x, ball_y = ball_location
        if self.random_ball_dir == 0 and ball_x + 1 < self.height and self.grid[
            ball_x + 1, ball_y] != 1:  # ball is going down and it's not agent
            ball_location = ball_location + np.array([1, 0])
        elif self.random_ball_dir == 1 and ball_y - 1 >= 0 and self.grid[ball_x, ball_y - 1] != 1:  # ball is going left
            ball_location = ball_location - np.array([0, 1])
        elif self.random_ball_dir == 2 and ball_x - 1 >= 0 and self.grid[ball_x - 1, ball_y] != 1:  # ball is going up
            ball_location = ball_location - np.array([1, 0])
        elif self.random_ball_dir == 3 and ball_y + 1 < self.width and self.grid[
            ball_x, ball_y + 1] != 1:  # ball is going right
            ball_location = ball_location + np.array([0, 1])
        return ball_location

    def check_ball_in_agent(self, ball_location):
        agent_x, agent_y = self.agent_states[0]
        all_ball_pos = [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x + 1, agent_y],
                        [agent_x, agent_y + 1]]
        return list(ball_location) in all_ball_pos

    def set_action_title(self):
        raise NotImplementedError

    def test_setup(self):
        self.use_ball = True
        self.count_test = 0
        self.num_tests = 30
        self.render_log = True
        # self.action = np.array([-1])
        self.reset_render_log()
        self.action_title = self.set_action_title()
        self.episode_max_length = 20

    def test_next(self):
        self.count_test += 1
        # self.action = np.array([-1])
        self.reset_render_log()

    def reset_render_log(self):
        self.render_agent = []
        self.render_ball = []
        self.render_target = []
        self.render_action = []
        self.render_reward = []

    def update_render_log(self):
        self.render_agent += [copy.deepcopy(self.agent_states)]
        self.render_ball += [copy.deepcopy(self.ball_location)]
        self.render_target += [copy.deepcopy(self.target_center)]
        self.render_action += [copy.deepcopy(self.action.flatten())]
        self.render_reward += [self.reward]


class PushTask(BaseTask):
    def __init__(self, do_training, config):
        super().__init__(do_training, config)
        self.goal_dim = task_dim['push']['goal']
        self.actor_goals = np.zeros((self.n_agent, self.goal_dim))
        self.action_dim = task_dim['push']['action']
        self.set_dimensions(self.action_dim, self.agent_state_dim, self.goal_dim)
        self.episode_max_length = 600
        self.success_value = 1.0

    def task_reset(self):
        self.grid[...] = 0.0

        self.random_agent_start()
        self.pre_agent_location = copy.deepcopy(self.agent_states[0])
        self.episode_step = 0
        self.task_complete = False
        self.shooting_a_goal = False
        self.last_value = 0.0
        self.reward = 0.0
        self.action[...] = -1

        self.ball_in_target = False

        self.random_ball_start()
        self.random_target_start()

        ''' Reset ball location'''
        self.ball_location = self.ball_start[:2]
        self.cal_pre_ball_loc()

        ''' Reset target ball distance'''
        self.target_direction_ball = self.target_center - self.ball_location
        self.pre_target_direction_ball = copy.deepcopy(self.target_direction_ball)

        if self.render_log:
            self.update_render_log()

    def task_reward(self):
        self.update_target_ball()
        self.ball_in_target = all(self.ball_location == self.target_center)
        agent_dist_imprv = 0
        if not self.contribute:
            agent_ball_dist = self.agent_states[0] - self.ball_location
            pre_agent_ball_dist = self.pre_agent_location - self.pre_ball_location
            agent_dist_imprv = np.linalg.norm(pre_agent_ball_dist, 1) - np.linalg.norm(agent_ball_dist, 1)
        return self.contribute * (0.05 * self.ball_dist_imprv + self.ball_in_target) + (
                1 - self.contribute) * 0.05 * agent_dist_imprv

    def set_action_title(self):
        return ['down', 'left', 'up', 'right', 'start']

    def do_action(self, a, i_agent):
        '''
        action 0: row+1 down
        action 1: col-1 left
        action 2: row-1 up
        action 3: col+1 right
        '''
        # a=int(input("action?"))
        agent_x, agent_y = self.agent_states[i_agent]
        self.pre_agent_location[...] = [agent_x, agent_y]
        controlling_ball = False
        # target or other agents
        if a == 0 and agent_x < self.height - 1 and self.grid[agent_x + 1, agent_y] not in [1, 3]:
            pushable = True
            if self.grid[agent_x + 1, agent_y] == 2:  # ball
                if agent_x + 2 < self.height and self.grid[agent_x + 2, agent_y] != 1:
                    self.random_ball_dir = 4
                    self.ball_location[0] += 1
                    self.grid[agent_x + 2, agent_y] = 2
                    controlling_ball = True
                else:
                    pushable = False
            if pushable:
                self.grid[agent_x, agent_y] = 0
                self.grid[agent_x + 1, agent_y] = 1
                self.agent_states[i_agent][0] += 1
        elif a == 2 and agent_x > 0 and self.grid[agent_x - 1, agent_y] not in [1, 3]:
            pushable = True
            if self.grid[agent_x - 1, agent_y] == 2:  # ball
                if agent_x - 1 > 0 and self.grid[agent_x - 2, agent_y] != 1:
                    self.random_ball_dir = 4
                    self.ball_location[0] -= 1
                    self.grid[agent_x - 2, agent_y] = 2
                    controlling_ball = True
                else:
                    pushable = False
            if pushable:
                self.grid[agent_x, agent_y] = 0
                self.grid[agent_x - 1, agent_y] = 1
                self.agent_states[i_agent][0] -= 1
        elif a == 3 and agent_y < self.width - 1 and self.grid[agent_x, agent_y + 1] not in [1, 3]:
            pushable = True
            if self.grid[agent_x, agent_y + 1] == 2:  # ball
                if agent_y + 2 < self.width and self.grid[agent_x, agent_y + 2] != 1:
                    self.random_ball_dir = 4
                    self.ball_location[1] += 1
                    self.grid[agent_x, agent_y + 2] = 2
                    controlling_ball = True
                else:
                    pushable = False
            if pushable:
                self.grid[agent_x, agent_y] = 0
                self.grid[agent_x, agent_y + 1] = 1
                self.agent_states[i_agent][1] += 1
        elif a == 1 and agent_y > 0 and self.grid[agent_x, agent_y - 1] not in [1, 3]:
            pushable = True
            if self.grid[agent_x, agent_y - 1] == 2:  # ball
                if agent_y - 1 > 0 and self.grid[agent_x, agent_y - 2] != 1:
                    self.random_ball_dir = 4
                    self.ball_location[1] -= 1
                    self.grid[agent_x, agent_y - 2] = 2
                    controlling_ball = True
                else:
                    pushable = False
            if pushable:
                self.grid[agent_x, agent_y] = 0
                self.grid[agent_x, agent_y - 1] = 1
                self.agent_states[i_agent][1] -= 1
        if controlling_ball:
            self.contribute = True
            # print("push ", self.contribute)
        else:
            self.contribute = False
            # print("push FREEEE", self.contribute)
            # self.free_ball_dynamics()
        self.ball_out_range = self.ball_location[0] == 0 or self.ball_location[0] == self.height - 1 or \
                              self.ball_location[1] == 0 or self.ball_location[1] == self.width - 1


class ThrowTask(BaseTask):
    def __init__(self, do_training, config):
        super().__init__(do_training, config)
        self.goal_dim = task_dim['throw']['goal']
        self.actor_goals = np.zeros((self.n_agent, self.goal_dim))
        self.action_dim = task_dim['throw']['action']  # throw up, down, left, right, wait
        self.set_dimensions(self.action_dim, self.agent_state_dim, self.goal_dim)
        self.episode_max_length = 10
        self.success_value = 1.0

        self.generate_target = self.random_target_start_align_ball
        self.generate_ball = self.generate_ball_train

    def generate_ball_train(self):
        self.random_ball_start_align_with_agent()

        self.random_ball_dir = self.rng.integers(0, 5)  # 'down', 'left', 'up', 'right','still'

    def task_reset(self):
        self.grid[...] = 0.0

        self.random_agent_start()
        self.pre_agent_location = copy.deepcopy(self.agent_states[0])
        self.episode_step = 0
        self.task_complete = False
        self.shooting_a_goal = False
        self.last_value = 0.0
        self.reward = 0.0
        self.action[...] = -1

        self.ball_in_target = False
        self.generate_ball()
        self.generate_target()

        ''' Reset ball location'''
        self.ball_location = self.ball_start[:2]
        self.cal_pre_ball_loc()

        ''' Reset target ball distance'''
        self.target_direction_ball = self.target_center - self.ball_location
        self.pre_target_direction_ball = self.target_center - self.pre_ball_location

        if self.render_log:
            self.update_render_log()

    def task_reward(self):
        self.update_target_ball()
        self.ball_in_target = all(self.ball_location == self.target_center)
        self.check_align_ball_target()
        return self.contribute * self.shooting_a_goal  # + self.ball_in_target

    def set_action_title(self):
        return ['throw down', 'throw left', 'throw up', 'throw right', 'wait', 'start']

    def check_valid_action(self, ball_location, agent_states, a):
        agent_x, agent_y = agent_states
        if a == 0:
            return list(ball_location) in [[agent_x + 1, agent_y], [agent_x, agent_y - 1], [agent_x, agent_y + 1]]
        elif a == 1:
            return list(ball_location) in [[agent_x + 1, agent_y], [agent_x - 1, agent_y], [agent_x, agent_y - 1]]
        elif a == 2:
            return list(ball_location) in [[agent_x - 1, agent_y], [agent_x, agent_y - 1], [agent_x, agent_y + 1]]
        elif a == 3:
            return list(ball_location) in [[agent_x + 1, agent_y], [agent_x - 1, agent_y], [agent_x, agent_y + 1]]
        else:
            return True

    def do_action(self, a, i_agent):
        '''
        action 0: row+1 throw down ->
            action available when ball loc is [agent_x+1,agent_y], [agent_x,agent_y-1], [agent_x,agent_y+1]
        action 1: col-1 left ->
            action available when ball loc is [agent_x+1,agent_y], [agent_x-1,agent_y], [agent_x,agent_y-1]
        action 2: row-1 up ->
            action available when ball loc is [agent_x-1,agent_y], [agent_x,agent_y-1], [agent_x,agent_y+1]
        action 3: col+1 right ->
            action available when ball loc is [agent_x+1,agent_y], [agent_x-1,agent_y], [agent_x,agent_y+1]
        action 4: wait
        '''
        agent_x, agent_y = self.agent_states[i_agent]
        self.pre_agent_location[...] = [agent_x, agent_y]
        self.contribute = False
        ball_x, ball_y = self.ball_location
        if a != 4:  # agent decides to do sth
            valid_action = self.check_valid_action(self.ball_location, self.agent_states[i_agent], a)
            if valid_action:
                next_ball_location = self.ball_location
            elif self.ball_move_count == 0:
                next_ball_location = self.cal_next_ball_loc(self.ball_location)
                self.ball_in_target = all(next_ball_location == self.target_center)
                if self.ball_in_target:
                    self.grid[self.ball_location[0], self.ball_location[1]] = 0
                    self.ball_location[...] = next_ball_location
                    self.grid[self.ball_location[0], self.ball_location[1]] = 2
                    self.ball_out_range = self.ball_location[0] == 0 or self.ball_location[0] == self.height - 1 or \
                                          self.ball_location[1] == 0 or self.ball_location[1] == self.width - 1
                    # print("throw", self.contribute)
                    return
                valid_action = self.check_valid_action(next_ball_location, self.agent_states[i_agent], a)
            else:
                valid_action = False
            if valid_action:
                self.contribute = True
                self.random_ball_dir = a
                self.grid[ball_x, ball_y] = 0
                next_ball_location = self.cal_next_ball_loc(next_ball_location)
                self.ball_location[...] = next_ball_location
                self.grid[self.ball_location[0], self.ball_location[1]] = 2
                # print("throw", self.contribute)
            # else:
            # print("throw FREEEE", self.contribute)
            # self.free_ball_dynamics()
        # else:
        # print("throw FREEEE", self.contribute)
        # self.free_ball_dynamics()
        self.ball_out_range = self.ball_location[0] == 0 or self.ball_location[0] == self.height - 1 or \
                              self.ball_location[1] == 0 or self.ball_location[1] == self.width - 1

    def test_setup(self):
        super().test_setup()
        self.generate_target = self.random_target_start_align_ball
        self.generate_ball = self.random_ball_start_align_with_agent


class CatchTask(BaseTask):
    def __init__(self, do_training, config):
        super().__init__(do_training, config)
        self.goal_dim = task_dim['catch']['goal']
        self.actor_goals = np.zeros((self.n_agent, self.goal_dim))
        self.action_dim = task_dim['catch']['action']  # go up, down, left, right, wait
        self.set_dimensions(self.action_dim, self.agent_state_dim, self.goal_dim)
        self.episode_max_length = 600
        self.success_value = 1.0

        self.generate_target = self.random_target_start
        self.generate_ball = self.generate_ball_train

    def generate_ball_train(self):
        self.random_ball_start_far_from_agent()
        self.random_ball_dir = self.rng.integers(0, 5)  # 'down', 'left', 'up', 'right','still'

    def task_reset(self):
        self.grid[...] = 0.0

        self.random_agent_start()
        self.pre_agent_location = copy.deepcopy(self.agent_states[0])
        self.episode_step = 0
        self.task_complete = False
        self.shooting_a_goal = False
        self.last_value = 0.0
        self.reward = 0.0
        self.action[...] = -1

        self.ball_in_target = False
        self.ball_in_agent = False
        self.generate_ball()
        self.generate_target()

        ''' Reset ball location'''
        self.ball_location = self.ball_start[:2]
        self.cal_pre_ball_loc()

        ''' Reset target ball distance'''
        self.target_direction_ball = self.target_center - self.ball_location
        self.pre_target_direction_ball = self.target_center - self.pre_ball_location

        if self.render_log:
            self.update_render_log()

    def set_action_title(self):
        return ['down', 'left', 'up', 'right', 'wait', 'start']

    def agent_move(self, a):
        agent_x, agent_y = self.agent_states[0]

        # check the step won't land in [1,3] target or other agents
        if a == 0 and agent_x < self.height - 1 and self.grid[agent_x + 1, agent_y] not in [1, 3]:
            self.grid[agent_x, agent_y] = 0
            self.grid[agent_x + 1, agent_y] = 1
            self.agent_states[0][0] += 1
        elif a == 2 and agent_x > 0 and self.grid[agent_x - 1, agent_y] not in [1, 3]:
            self.grid[agent_x, agent_y] = 0
            self.grid[agent_x - 1, agent_y] = 1
            self.agent_states[0][0] -= 1
        elif a == 3 and agent_y < self.width - 1 and self.grid[agent_x, agent_y + 1] not in [1, 3]:
            self.grid[agent_x, agent_y] = 0
            self.grid[agent_x, agent_y + 1] = 1
            self.agent_states[0][1] += 1
        elif a == 1 and agent_y > 0 and self.grid[agent_x, agent_y - 1] not in [1, 3]:
            self.grid[agent_x, agent_y] = 0
            self.grid[agent_x, agent_y - 1] = 1
            self.agent_states[0][1] -= 1

    def do_action(self, a, i_agent):
        '''
        action 0: down
        action 1: col-1 left
        action 2: row-1 up
        action 3: col+1 right
        action 4: wait
        '''
        # a=int(input("action?"))
        agent_x, agent_y = self.agent_states[i_agent]
        self.pre_agent_location[...] = [agent_x, agent_y]
        self.contribute = False
        ball_in_agent = self.check_ball_in_agent(self.ball_location)
        if ball_in_agent:
            if self.random_ball_dir != 4:
                self.contribute = True
            self.random_ball_dir = 4
            # print("catch", self.contribute)
        else:
            # if self.random_ball_dir!=4:
            next_ball_location = self.cal_next_ball_loc(self.ball_location)
            # next_ball_in_agent = self.check_ball_in_agent(next_ball_location)
            if self.random_ball_dir != 4 and self.check_ball_in_agent(next_ball_location):
                self.grid[self.ball_location[0], self.ball_location[1]] = 0
                self.ball_location[...] = next_ball_location
                self.grid[self.ball_location[0], self.ball_location[1]] = 2
                self.contribute = True
                self.random_ball_dir = 4
                self.ball_in_agent = True
                # print("catch", self.contribute)
            else:
                self.agent_move(a)
                next_ball_in_agent = self.check_ball_in_agent(next_ball_location)
                if next_ball_in_agent:
                    self.grid[self.ball_location[0], self.ball_location[1]] = 0
                    self.ball_location[...] = next_ball_location
                    self.grid[self.ball_location[0], self.ball_location[1]] = 2
                    self.contribute = True
                    self.random_ball_dir = 4
                    self.ball_in_agent = True
                    # print("catch", self.contribute)
                else:
                    next_next_ball_location = self.cal_next_ball_loc(next_ball_location)
                    next_next_ball_in_agent = self.check_ball_in_agent(next_next_ball_location)
                    if next_next_ball_in_agent:
                        self.grid[self.ball_location[0], self.ball_location[1]] = 0
                        self.ball_location[...] = next_next_ball_location
                        self.grid[self.ball_location[0], self.ball_location[1]] = 2
                        self.contribute = True
                        self.random_ball_dir = 4
                        self.ball_in_agent = True
                        # print("catch", self.contribute)
                    # else:
                    #     print("catch FREEEE", self.contribute)
                    # self.free_ball_dynamics()
        self.ball_out_range = self.ball_location[0] == 0 or self.ball_location[0] == self.height - 1 or \
                              self.ball_location[1] == 0 or self.ball_location[1] == self.width - 1

        #     if self.random_ball_dir!=4:
        #         next_ball_location = self.cal_next_ball_loc(self.ball_location)
        #         next_ball_in_agent = self.check_ball_in_agent(next_ball_location)
        #         if next_ball_in_agent:
        #             self.grid[self.ball_location[0], self.ball_location[1]] = 0
        #             self.ball_location[...] = next_ball_location
        #             self.grid[self.ball_location[0], self.ball_location[1]] = 2
        #             self.contribute = True
        #             self.random_ball_dir = 4
        #         else:
        #             self.agent_move(a)
        #             check again if ball in agent
        #             if ball_in_agent:
        #                 self.contribute=True
        #                 self.random_ball_dir=4
        #     else:
        #         self.agent_move(a)
        #
        #     self.free_ball_dynamics()
        #
        #     self.ball_in_agent=self.check_ball_in_agent(self.ball_location)
        #     if self.ball_in_agent:
        #         self.contribute = True
        #         self.random_ball_dir = 4
        # else:
        #     self.contribute = False
        #     self.random_ball_dir = 4

    def task_reward(self):
        # print("fix ball_align_target")
        # exit()
        self.update_target_ball()
        self.ball_in_target = all(self.ball_location == self.target_center)
        self.check_align_ball_target()  # orignally has self.align_ball_target can only be true is ball_in_agent is also true

        agent_dist_imprv = 0
        if not self.contribute:
            agent_ball_dist = self.agent_states[0] - self.ball_location
            pre_agent_ball_dist = self.pre_agent_location - self.pre_ball_location
            agent_dist_imprv = np.linalg.norm(pre_agent_ball_dist, 1) - np.linalg.norm(agent_ball_dist, 1)
        # print(self.ball_location,self.pre_ball_location)
        # print(self.contribute * (self.align_ball_target + self.ball_in_agent) + (
        #                 1 - self.contribute) * 0.05 * agent_dist_imprv)
        # print((self.align_ball_target + self.ball_in_agent), agent_dist_imprv)
        # input(" ")
        return self.contribute * (self.align_ball_target + self.ball_in_agent) + (
                1 - self.contribute) * 0.05 * agent_dist_imprv

    def task_done(self):
        basic_done = super().task_done()
        # ball_in_agent_control
        if self.ball_in_agent:
            if self.align_ball_target:
                self.last_value = self.success_value
                self.task_complete = True
            else:
                self.last_value = 0.0
            return True
        return basic_done
