from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import mtl
import random

class MultiStlEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, cells, spec=None, seed=None, random_start=False):
        self.cells = cells
        self.height, self.width, _ = cells.shape

        self.agent_pos_list = []
        for j in range(self.height):
            for i in range(self.width):
                if self.cells[j, i, 0] == 10:
                    self.agent_pos_list.append(np.array([i,j,self.cells[j, i, 2]]))
                    self.cells[j, i, :] = np.array([0, 0, 0])

        super().__init__(
            height=self.height,
            width=self.width,
            max_steps=4*(self.height+self.width),
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

        self.goal_reward = 1
        self.failure_reward = 0
        self.step_penalty = -0.01
        self.reward_range = (self.max_steps*self.step_penalty+self.failure_reward,
                             self.goal_reward)

        self.path = []
        self.lavas = []

        if spec:
            self.phi = mtl.parse(spec)
            self.reward_size = 2
        else:
            self.phi = None
            self.reward_size = 1

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        self.lavas = []

        a_pos = random.choice(self.agent_pos_list)
        self.agent_pos = (a_pos[0], a_pos[1])
        self.agent_dir = a_pos[2]

        for j in range(self.height):
            for i in range(self.width):
                obj = self._parse_object(self.cells[j,i,:])
                if obj=='agent':
                    # rearranged things so there should never be an 'agent' cell at this point
                    raise ValueError
                elif obj==None:
                    pass
                else:
                    self.put_obj(obj, i, j)

                if type(obj) == Lava or \
                    (type(obj) == Wall and obj.color == 'red'):
                    self.lavas.append((i,j))

        self.mission = (
            "avoid the lava and get to the green goal square"
        )

    def _parse_object(self, cell_desc):
        cell_type, cell_color_ID, cell_state = cell_desc

        obj_type = IDX_TO_OBJECT[cell_type]
        cell_color = IDX_TO_COLOR[cell_color_ID]

        if obj_type == 0:
            pass
        elif obj_type == 'agent':
            return 'agent'
        elif obj_type == 'wall':
            return Wall(cell_color)
        elif obj_type == 'floor':
            return Floor(cell_color)
        elif obj_type == 'door':
            return Door(cell_color)
        elif obj_type == 'key':
            return Key(cell_color)
        elif obj_type == 'ball':
            return Ball(cell_color)
        elif obj_type == 'box':
            return Box(cell_color)
        elif obj_type == 'goal':
            return Goal()
        elif obj_type == 'lava':
            return Lava()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        self.path.append(self.agent_pos)

        if self.phi:
            reward = np.array([0.0,0.0])
        else:
            reward = np.array([0.0])

        if done:
            if tuple(self.agent_pos) in [c.cur_pos for c in self.grid.grid if c and c.type=='goal']:
                if self.phi:  # this is the "STL" implemetation
                    reward[1] = self.phi (self.get_signals())
                reward[0] = self.goal_reward
                #print('goal!')
            elif tuple(self.agent_pos) in [c.cur_pos for c in self.grid.grid if c and c.type=='lava']:
                if self.phi:
                    reward[1] = self.failure_reward
                reward[0] = self.failure_reward
                #print('lava!')

        else:
            reward[0] = self.step_penalty
            #reward[1] = self.step_penalty

        if not self.phi:
            reward = reward.item()
        else:
            reward = reward #reward[1].item()

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()

        self.path = []
        return obs


    def get_signals(self):
        data = {
            'a' : [(t,min([((p[0]-l[0])**2 + (p[1]-l[1])**2)**0.5 for l in self.lavas]))
                   for (t,p) in enumerate(self.path)]
        }
        return data



class TripleCrossingEnv(MultiStlEnv):
    def __init__(self):
        p = [2,5,0]  # perimiter (it's a wall,but it's than the "wall")
        w = [9,0,0]  # "wall" (9 = lava,2 = wall)
        a = [10,0,0]  # agent
        g = [8,0,0]  # goal
        f = [0,0,0]  # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,a,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            #                                      #
            [p,f,f,f,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,f,w,w,w,w,f,f,p],
            #                                      #
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])
        phi = '(G a)'
        #phi = None
        super().__init__(cells=grid_cells, spec=phi)

class TripleCrossingEnv_Random(MultiStlEnv):
    def __init__(self):
        p = [2, 5, 0]  # perimiter (it's a wall, but it's than the "wall")
        w = [9, 0, 0]  # "wall" (9 = lava, 2 = wall)
        a = [10, 0, 0]  # agent
        g = [8, 0, 0]  # goal
        f = [0, 0, 0]  # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            #                                      #
            [p,f,f,f,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,f,w,w,w,w,f,f,p],
            #                                      #
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])

        phi = '(G a)'
        #phi = None
        super().__init__(cells=grid_cells, spec=phi)

class TripleCrossingWallEnv(MultiStlEnv):
    def __init__(self):
        p = [2, 5, 0]  # perimiter (it's a wall, but it's than the "wall")
        w = [2, 0, 0]  # "wall" (9 = lava, 2 = wall)
        a = [10, 0, 0]  # agent
        g = [8, 0, 0]  # goal
        f = [0, 0, 0]  # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,a,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            #                                      #
            [p,f,f,f,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,f,w,w,w,w,f,f,p],
            #                                      #
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])

        phi = '(G a)'
        #phi = None
        super().__init__(cells=grid_cells, spec=phi)

class TripleCrossingWallEnv_Random(MultiStlEnv):
    def __init__(self):
        p = [2, 5, 0]  # perimiter (it's a wall, but it's than the "wall")
        w = [2, 0, 0]  # "wall" (9 = lava, 2 = wall)
        a = [10, 0, 0]  # agent
        g = [8, 0, 0]  # goal
        f = [0, 0, 0]  # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            #                                      #
            [p,f,f,f,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,f,w,w,w,w,f,f,p],
            #                                      #
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])

        phi = '(G a)'
        #phi = None
        super().__init__(cells=grid_cells, spec=phi)

class TripleCrossingNarrowEnv(MultiStlEnv):
    def __init__(self):
        p = [2,5,0]  # perimiter (it's a wall,but it's than the "wall")
        w = [9,0,0]  # "wall" (9 = lava,2 = wall)
        a = [10,0,0]  # agent
        g = [8,0,0]  # goal
        f = [0,0,0]  # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,f,f,f,f,f,f,f,f,a,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            #                                      #
            [p,f,f,f,w,w,w,w,w,f,w,w,w,w,f,f,p],
            #                                      #
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])
        phi = '(G a)'
        #phi = None
        super().__init__(cells=grid_cells, spec=phi)

class TripleCrossingNarrowEnv_Random(MultiStlEnv):
    def __init__(self):
        p = [2, 5, 0]  # perimiter (it's a wall, but it's than the "wall")
        w = [9, 0, 0]  # "wall" (9 = lava, 2 = wall)
        a = [10, 0, 0]  # agent
        g = [8, 0, 0]  # goal
        f = [0, 0, 0]  # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            #                                      #
            [p,f,f,f,w,w,w,w,w,f,w,w,w,w,f,f,p],
            #                                      #
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])

        phi = '(G a)'
        #phi = None
        super().__init__(cells=grid_cells, spec=phi)


register(
    id='MiniGrid-TripleCrossing-v0',
    entry_point='gym_minigrid.envs:TripleCrossingEnv'
)

register(
    id='MiniGrid-TripleCrossing-Random-v0',
    entry_point='gym_minigrid.envs:TripleCrossingEnv_Random'
)

register(
    id='MiniGrid-TripleCrossing-Walls-v0',
    entry_point='gym_minigrid.envs:TripleCrossingWallsEnv'
)

register(
    id='MiniGrid-TripleCrossing-Walls-Random-v0',
    entry_point='gym_minigrid.envs:TripleCrossingWallsEnv_Random'
)

register(
    id='MiniGrid-TripleCrossing-Narrow-v0',
    entry_point='gym_minigrid.envs:TripleCrossingNarrowEnv'
)

register(
    id='MiniGrid-TripleCrossing-Narrow-Random-v0',
    entry_point='gym_minigrid.envs:TripleCrossingNarrowEnv_Random'
)
