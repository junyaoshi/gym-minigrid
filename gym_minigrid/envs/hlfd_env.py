from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class HLfDEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, agent_view_size=3, obstacle_type=Lava, obstacle_see_behind=False):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.agent_view_size = agent_view_size
        self.obstacle_type = obstacle_type
        self.obstacle_see_behind = obstacle_see_behind
        self.colors = set(COLOR_NAMES)

        # max and min dsitance between two obstacles
        self._obstacle_min_gap = 4
        self._obstacle_max_gap = 6

        # Specify x indices of three obstacles
        self.obs_1_x = np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)
        self.obs_2_x = self.obs_1_x + np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)
        self.obs_3_x = self.obs_2_x + np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)

        self.obs_x_indices = [self.obs_1_x, self.obs_2_x, self.obs_3_x]
        self.obs_generation_funcs = [self._generate_obs_avoid, self._generate_obs_move_away,
                                     self._generate_obs_move_into]
        assert len(self.obs_x_indices) == len(self.obs_generation_funcs)

        self.width = self.obs_3_x + np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)
        self.height = 9
        self.max_steps = np.inf

        super().__init__(height=self.height, width=self.width, max_steps=self.max_steps)

    @property
    def front_front_pos(self):
        """
        Get the position of the cell that is in front of the cell in front of the agent
        """
        return self.agent_pos + self.dir_vec * 2

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Generate obstacles
        np.random.shuffle(self.obs_x_indices)
        np.random.shuffle(self.obs_generation_funcs)
        for i in range(len(self.obs_x_indices)):
            x = self.obs_x_indices[i]
            obs_gen_func = self.obs_generation_funcs[i]
            obs_gen_func(x)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent(top=(1, 1), size=(1, height - 2))

        # Randomize the goal position
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
            self.goal_pos = self._goal_default_pos
        else:
            self.goal_pos = self.place_obj(Goal(), top=(self.width - 2, 1), size=(1, height - 2))

        self.mission = 'Reach the goal'

    def _generate_door(self):
        doorColor = self._rand_elem(sorted(self.colors))
        return Door(doorColor)

    def _generate_ball(self):
        ballColor = self._rand_elem(sorted(self.colors))
        return Ball(ballColor)

    def _generate_movable_block(self):
        blockColor = self._rand_elem(sorted(self.colors))
        return MovableBlock(blockColor)

    def _generate_obs_avoid(self, x):
        self.grid.vert_wall(x=x, y=1, length=self.height - 2, obj_type=self.obstacle_type,
                            see_behind=self.obstacle_see_behind)
        holePos = (x, self._rand_int(1, self.height - 1))
        self.grid.set(*holePos, None)

    def _generate_obs_door(self, x):
        self.grid.vert_wall(x=x, y=1, length=self.height - 2, obj_type=self.obstacle_type,
                            see_behind=self.obstacle_see_behind)
        entryDoor = self._generate_door()
        doorPos = (x, self._rand_int(1, self.height - 1))
        self.grid.set(*doorPos, entryDoor)

    def _generate_obs_move_away(self, x):
        self.grid.vert_wall(x=x, y=1, length=self.height - 2, obj_type=self.obstacle_type,
                            see_behind=self.obstacle_see_behind)
        y = self._rand_int(2, self.height - 2)
        holePos = (x, y)
        block = self._generate_movable_block()
        blockPos = (x - 1, y)
        self.grid.set(*holePos, None)
        self.grid.set(*blockPos, block)

    def _generate_obs_move_into(self, x):
        self.grid.vert_wall(x=x, y=1, length=self.height - 2, obj_type=self.obstacle_type,
                            see_behind=self.obstacle_see_behind)
        y = self._rand_int(1, self.height - 1)
        block = self._generate_movable_block()
        blockPos = (x - 1, y)
        self.grid.set(*blockPos, block)

    def _generate_obs_ball(self, x):
        self.grid.vert_wall(x=x, y=1, length=self.height - 2, obj_type=self.obstacle_type,
                            see_behind=self.obstacle_see_behind)
        y = self._rand_int(1, self.height - 1)
        holePos = (x, y)
        ball = self._generate_ball()
        ballPos = (x - 1, y)
        self.grid.set(*holePos, None)
        self.grid.set(*ballPos, ball)

    def _reward(self):
        return 1

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        self.agent_view_size = 5
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            # 'mission': self.mission,
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
        }

        return obs

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'movable_block':

                # Get the position in front front of the agent
                fwd_fwd_pos = self.front_front_pos

                # Get the contents of the cell in front front of the agent
                fwd_fwd_cell = self.grid.get(*fwd_fwd_pos)

                if fwd_fwd_cell is None:
                    self.grid.set(*fwd_pos, None)
                    self.grid.set(*fwd_fwd_pos, fwd_cell)
                    self.agent_pos = fwd_pos
                elif fwd_fwd_cell.type == 'lava':
                    self.grid.set(*fwd_pos, None)
                    self.grid.set(*fwd_fwd_pos, None)
                    self.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def reset(self):
        # Specify x indices of three obstacles
        self.obs_1_x = np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)
        self.obs_2_x = self.obs_1_x + np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)
        self.obs_3_x = self.obs_2_x + np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)

        self.obs_x_indices = [self.obs_1_x, self.obs_2_x, self.obs_3_x]

        self.width = self.obs_3_x + np.random.randint(self._obstacle_min_gap, self._obstacle_max_gap + 1)

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs


register(
    id='MiniGrid-HLfD-v0',
    entry_point='gym_minigrid.envs:HLfDEnv'
)
