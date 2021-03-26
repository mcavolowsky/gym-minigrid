from gym_minigrid.minigrid import *
from gym_minigrid.register import register



class MultiStlEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, cells, seed=None):
        self.cells = cells
        self.height, self.width, _ = cells.shape

        super().__init__(
            height=self.height,
            width=self.width,
            max_steps=4*(self.height+self.width),
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

        self.path = []
        self.lavas = []

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        self.lavas = []

        for j in range(self.height):
            for i in range(self.width):
                obj = self._parse_object(self.cells[j,i,:])
                if obj=='agent':
                    self.agent_pos = (i, j)
                    self.agent_dir = self.cells[j,i,2]
                elif obj==None:
                    pass
                else:
                    self.put_obj(obj, i, j)

                if type(obj) == Lava:
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

        if done:
            if tuple(self.agent_pos) in [c.cur_pos for c in self.grid.grid if c and c.type=='goal']:
                reward = 1
                print('goal!')
            elif tuple(self.agent_pos) in [c.cur_pos for c in self.grid.grid if c and c.type=='lava']:
                reward = -1
                print('lava!')
            if False: # this is the "STL" implemetation
                reward = min([(x_a-x_l)**2+(y_a-y_l)**2
                              for (x_a, y_a) in self.path
                              for (x_l,y_l) in self.lavas])**0.5
        else:
            reward = -0.01

        return obs, reward, done, info

    def reset(self):
        obs = super().reset()

        self.path = []
        return obs


class TripleCrossingEnv(MultiStlEnv):
    def __init__(self):
        p = [2, 5, 0]       # perimiter (it's a wall, but it's than the "wall")
        w = [2, 0, 0]       # "wall" (9 = lava, 2 = wall)
        a = [10, 0, 0]      # agent
        g = [8, 0, 0]       # goal
        f = [0, 0, 0]       # floor

        grid_cells = np.array([
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,a,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
#            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
#            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
#            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],

            [p,f,f,f,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,f,w,w,w,w,f,f,p],

#            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
#            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
#            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,p],
            [p,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,g,f,f,f,f,f,f,p],
            [p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p,p]
        ])
        super().__init__(cells=grid_cells)

register(
    id='MiniGrid-TripleCrossing-v0',
    entry_point='gym_minigrid.envs:TripleCrossingEnv'
)
