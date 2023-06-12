import math
import numpy as np
import matplotlib.pyplot as plt

def perturb(data: np.ndarray, scale: float):
    return data + np.random.normal(scale=scale, size=data.shape)


class Scenario:
    def __init__(self, num_agent: int, radius: float = .1, seed: int = 42):
        np.random.seed(seed)
        self.num_agent = num_agent
        self.radius = radius
        self.initial_conditions = np.zeros((num_agent, 4))
        self.goals = np.zeros((num_agent, 2))

    def color(self, i):
        return f"C{i}"

    def plot_init(self, show: bool = True):
        fig, ax = plt.subplots()
        ax.set_xlim([-70., 70.])
        ax.set_ylim([-70., 70.])
        for i in range(self.num_agent):
            ax.add_patch(plt.Circle((self.initial_conditions[i, 0], self.initial_conditions[i, 1]),
                                    radius=self.radius, color=self.color(i)))
            ax.arrow(x=self.initial_conditions[i, 0],
                     y=self.initial_conditions[i, 1],
                     dx=np.cos(self.initial_conditions[i, 3]) * self.radius * 2,
                     dy=np.sin(self.initial_conditions[i, 3]) * self.radius * 2)

        if show:
            plt.show()

        return fig, ax

class CircularScenario(Scenario):
    def __init__(self, num_agent: int, initial_dist: float,
                 pos_noise: float = .2):
        super().__init__(num_agent)

        # construct initial conditions as a circle around the origin
        theta = np.linspace(-np.pi, np.pi, num_agent, endpoint=False)
        self.initial_conditions[:, 0] = np.cos(theta) * initial_dist
        self.initial_conditions[:, 1] = np.sin(theta) * initial_dist

        # construct goals to be opposite to initial condition
        self.goals = -self.initial_conditions

        # add some random noise to the initial condition
        self.initial_conditions[:, :2] = perturb(self.initial_conditions[:, :2], scale=pos_noise)


class StraightScenario(Scenario):
    def __init__(self, num_agent: int, travel_dist: float, spacing: float,
                 pos_noise: float = 3., ang_noise: float = math.radians(30)):
        super().__init__(num_agent)

        # construct initial conditions as aligned horizontally pointing upwards
        total_spacing = spacing * num_agent
        self.initial_conditions[:, 0] = np.linspace(-total_spacing / 2, total_spacing / 2, num_agent)
        self.initial_conditions[:, 1] = -travel_dist
        self.initial_conditions[:, 3] = np.pi / 2

        # construct goals to shuffle horizontal order
        x_init = self.initial_conditions[:, 0].copy()
        np.random.shuffle(x_init)
        self.goals[:, 0] = x_init
        self.goals[:, 1] = travel_dist

        # add some random noise to the initial condition
        self.initial_conditions[:, 1] = perturb(self.initial_conditions[:, 1], scale=pos_noise)
        self.initial_conditions[:, 3] = perturb(self.initial_conditions[:, 3], scale=ang_noise)


class ExchangeScenario(Scenario):
    def __init__(self, num_agent: int, map_size: float = 100.,
                 pos_noise: float = 3., ang_noise: float = math.radians(30)):
        super().__init__(num_agent)

        # construct initial conditions as paired agents randomly placed in a map
        assert num_agent % 2 == 0, "must use even number of agents for exchange scenarios"
        while True:
            init_xy = np.random.uniform(-map_size / 2, map_size / 2, size=(num_agent, 2))
            if self._init_is_valid(init_xy):
                break
        self.initial_conditions[:, :2] = init_xy
        self.initial_conditions[:, 3] = np.random.uniform(-np.pi, np.pi, size=num_agent)

        # construct goals from exchanging positions from initial conditions
        half_num_agent = num_agent // 2
        self.goals[:half_num_agent] = self.initial_conditions[half_num_agent:, :2]
        self.goals[half_num_agent:] = self.initial_conditions[:half_num_agent, :2]

    def color(self, i):
        return f"C{i % (self.num_agent // 2)}"

    def _init_is_valid(self, init_xy: np.ndarray) -> bool:
        dist_mat = np.linalg.norm(init_xy[None] - init_xy[:, None], axis=-1)
        return dist_mat[dist_mat > 0].min() > 4 * self.radius


TEST_SCENARIOS = [
    CircularScenario(2, 2.),
    CircularScenario(3, 2.),
    CircularScenario(4, 2.),
    CircularScenario(6, 2.),
    CircularScenario(8, 2.),
    CircularScenario(16, 2.),
    #StraightScenario(3, 50., 12.),
    #StraightScenario(4, 50., 12.),
    #StraightScenario(6, 50., 12.),
    #StraightScenario(10, 50., 12.),
    #ExchangeScenario(2, 100.),
    #ExchangeScenario(4, 100.),
    #ExchangeScenario(6, 100.),
    #ExchangeScenario(8, 100.),
    #ExchangeScenario(16, 100.),
]
