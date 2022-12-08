import environment
from utils import *

GAMMA = 0.99  # decrease rate
ETA = 0.7  # learning rate

NUM_DIGITIZED = 5
R_SPACE = (0.6, 1.5, 4, 10)
ACC_SPACE = (-0.8, -0.5, -0.3, 0, 0.4, 0.8, 1)

RADAR_BINS = (0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5)
RADAR_DIRS = (np.pi / 3, np.pi / 6, 0, -np.pi / 6, -np.pi / 3)


def bins(clip_min, clip_max, num=NUM_DIGITIZED):
    return np.linspace(clip_min, clip_max, num + 1)[1: -1]


class Brain:

    def __init__(self,
                 action_r_space,
                 action_acc_space,
                 path_width,
                 radar_dirs,
                 num_digitize=NUM_DIGITIZED):
        self.gamma = GAMMA
        self.eta = ETA

        self.action_space_r = [i * path_width for i in action_r_space] + \
                              [0] + \
                              [-i * path_width for i in reversed(action_r_space)]
        self.action_space_acc = action_acc_space

        self.path_width = path_width

        self.num_r_actions = len(self.action_space_r)
        self.num_acc_actions = len(self.action_space_acc)
        self.num_actions = self.num_acc_actions * self.num_r_actions

        self.num_radar_states = (len(RADAR_BINS) + 1) ** len(RADAR_DIRS)
        self.num_states = self.num_radar_states * num_digitize

        self.q_table = np.random.uniform(0, 1, (self.num_states, self.num_actions))

        self.radar_bins = np.array(RADAR_BINS)
        self.radar_directions = radar_dirs

        self.speed_bins = bins(0.0, 32, num_digitize)

    def digitize_state(self, observation):
        """
        observation: distances to the edges detected by the radars + 'car.velocity'
        """
        radar_observation, velocity = observation[0], observation[1]

        radar_digitized = np.digitize(radar_observation / self.path_width, self.radar_bins)
        speed_digitized = np.digitize(velocity, self.speed_bins)

        radar_state = sum([x * (len(self.radar_bins) + 1) ** i for i, x in enumerate(radar_digitized)])
        speed_state = speed_digitized * self.num_radar_states

        return radar_state + speed_state

    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        max_q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
                                      self.eta * (reward + self.gamma * max_q_next - self.q_table[state, action])

    def get_action(self, observation, episode=0, training=True):
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode / 100 + 1)) + 0.02

        if training and epsilon > np.random.uniform(0, 1):
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table[state][:])

        return action

    def set_eta_gamma(self, eta=ETA, gamma=GAMMA):
        self.eta = eta
        self.gamma = gamma

    def load_q_tabel(self, file_path):
        try:
            table = np.load(file_path)
        except (FileNotFoundError, FileExistsError):
            print("File Not Exists.")
            return False

        if table.shape == self.q_table.shape:
            self.q_table = table
            return True
        return False

    def save_q_tabel(self, label="best"):
        np.save(label, self.q_table)


class Agent(Brain):

    def __init__(self,
                 action_r_space=R_SPACE,
                 action_acc_space=ACC_SPACE,
                 radar_dirs=RADAR_DIRS,
                 path_width=environment.PATH_WIDTH,
                 k=environment.k,
                 f=environment.f,
                 miu=environment.miu,
                 dt=environment.dt):
        super(Agent, self).__init__(action_r_space, action_acc_space, path_width, radar_dirs)

        self.k = k
        self.f = f
        self.miu = miu
        self.dt = dt

        self.position = np.array([0, 0])
        self.direction = np.array([1, 0])
        self.velocity = 0
        self.acceleration = 0

        self.hist_v = []
        self.hist_pos = []
        self.hist_dir = []

    def step_next(self, observation, episode=0, training=True):
        action = self.get_action(observation, episode, training)
        action_acc, action_r, action = self.decide_action(action)
        self.run_one_step(action_acc, action_r)
        self.store_hist()
        return action

    def decide_action(self, action):
        action_acc_index = action % self.num_acc_actions
        action_r_index = action // self.num_acc_actions

        return self.action_space_acc[action_acc_index], self.action_space_r[action_r_index], action

    def run_one_step(self, acc, radius):
        v0 = self.velocity
        a = self.accelerate(acc, v0)

        vt = v0 + a / 2

        self.velocity += a * self.dt
        new_position, new_dir = self.drive(self.position, radius, vt, self.direction, self.dt)
        self.position = new_position
        self.direction = new_dir
        self.acceleration = a

    def accelerate(self, acc, velocity):
        a = self.k * acc + self.miu * velocity ** 2 + self.f
        if velocity < 1e-3 and a < 0:
            return 0
        else:
            return a

    def get_pos_dir_v(self):
        return self.position, self.direction, self.velocity

    def set_pos_dir_v(self, p=None, direction=None, v=None):
        if p is not None and p.shape == self.position.shape:
            self.position = p
        if direction is not None and direction.shape == self.direction.shape:
            self.direction = direction
        if v is not None:
            self.velocity = v
        self.store_hist()

    def store_hist(self):
        self.hist_v.append(self.velocity)
        self.hist_pos.append(self.position)
        self.hist_dir.append(self.direction)

    def clear_hist(self):
        self.hist_v = []
        self.hist_pos = []
        self.hist_dir = []

    def get_hist(self):
        pos = np.array(self.hist_pos)
        drt = np.array(self.hist_dir)
        v = np.array(self.hist_v).reshape((-1, 1))
        return np.hstack([pos, drt, v])

    @staticmethod
    def drive(p, radius, velocity, direction, dt):
        direction = dir_e(direction)
        if radius == 0:
            return p + direction * velocity * dt, direction

        theta = velocity * dt / radius

        d_tau = radius * np.sin(theta)
        d_n = radius * (1 - np.cos(theta))
        dp = direction * d_tau + normal_2d(direction) * d_n

        dir_theta = dir2theta(direction)
        new_theta = dir_theta + theta
        new_dir = np.array([np.cos(new_theta), np.sin(new_theta)])

        return p + dp, new_dir
