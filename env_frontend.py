import gym, requests
import numpy as np

from time import sleep, time
from common.data_utils import load_signals, load_trajectory, adjust_indexes
from common.server_utils import post_signals, get_state, set_action

class MantsinenBasic(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, mvs, signal_csv, identical_input_signals, trajectory_data, server_url, frequency, dt, use_signals=False, binary_signals=[0]):

        # init params

        self.mvs = mvs
        self.id = None
        self.server = server_url
        self.act_freq = frequency
        self.dt = dt
        self.use_signals = use_signals
        self.binary_signals = binary_signals

        # load signals and their limits

        self.signals, mins, maxs = {}, {}, {}
        for key in signal_csv.keys():
            values, xmin, xmax = load_signals(signal_csv[key])
            self.signals[key] = values
            mins[key] = xmin
            maxs[key] = xmax

        # deal with identical signals

        obs_input_idx, self.act_idx = adjust_indexes(self.signals, identical_input_signals)
        self.obs_index = np.append(obs_input_idx, np.arange(len(self.signals['input']), len(self.signals['input']) + len(self.signals['output'])))

        # dimensions and standardization coefficients

        rew_dim = len(self.signals['reward'])
        obs_dim = 4 * rew_dim
        self.rew_min = np.array(mins['reward'])
        self.rew_max = np.array(maxs['reward'])
        obs_min = np.hstack([self.rew_min, self.rew_min, self.rew_min, self.rew_min])
        obs_max = np.hstack([self.rew_max, self.rew_max, self.rew_max, self.rew_max])
        self.obs_input_output_min = np.hstack([np.array(mins['input'])[obs_input_idx], mins['output']])
        self.obs_input_output_max = np.hstack([np.array(maxs['input'])[obs_input_idx], maxs['output']])
        if use_signals:
            obs_dim += len(self.obs_index)
            obs_min = np.append(obs_min, self.obs_input_output_min)
            obs_max = np.append(obs_max, self.obs_input_output_max)
        act_dim = len(obs_input_idx)
        self.act_min, self.act_max = np.array(mins['input'])[obs_input_idx], np.array(maxs['input'])[obs_input_idx]

        # extract data from the trajectory

        self.rew_xyz = trajectory_data[:, :rew_dim]
        self.rew_t = trajectory_data[:, -1]

        # calculate parameters for reward scaling

        dt = np.linalg.norm(self.rew_t[:, None, None] - self.rew_t[None, :, None], axis=-1)
        dx = np.linalg.norm(self.rew_xyz[:, None, :] - self.rew_xyz[None, :, :], axis=-1)
        self.dx_max = np.max(dx)
        self.dt_max = np.max(dt)
        self.dxdt_max = np.nanmax(dx / dt)

        # set spaces

        self.observation_space = gym.spaces.Box(low=obs_min, high=obs_max, shape=(obs_dim,), dtype=np.float)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,), dtype=np.float)

        # tmp

        self.actions = []

    def reset(self):
        print(len(self.actions))
        with open('tmp/actions_in_{0}.txt'.format(self.id), 'w') as f:
            for action in self.actions:
                line = ','.join([str(item) for item in action])
                f.write(line + '\n')
        post_signals(self.server, self.id, self.signals)
        input_output_obs, reward_components = [], []
        while len(input_output_obs) == 0 or len(reward_components) == 0:
            sleep(self.act_freq)
            input_output_obs, reward_components = self._get_state()
        t_now = time()
        real_xyz = self._std_vector(reward_components, self.rew_min, self.rew_max)
        self.start_time = t_now
        t_elapsed_from_the_start = t_now - self.start_time
        target_xyz = self._predict_xyz(t_elapsed_from_the_start)
        next_target_xyz = self._predict_xyz(t_elapsed_from_the_start + self.dt)
        obs = np.hstack([real_xyz, target_xyz, real_xyz, next_target_xyz])
        if self.use_signals:
            obs = np.append(obs, self._std_vector(np.array(input_output_obs)[self.obs_index], self.obs_input_output_min, self.obs_input_output_max))
        self.last_reward_components = np.array(reward_components)
        self.last_target_components = np.array(target_xyz)
        self.last_time = t_now
        return obs

    def step(self, action):
        action = self._std_vector(action, self.action_space.low, self.action_space.high)
        action = self._orig_vector(action, self.act_min, self.act_max)
        self._set_action(action[self.act_idx])
        sleep(self.act_freq)
        input_output_obs, reward_components = self._get_state()
        t_now = time()
        t_elapsed_from_start = t_now - self.start_time
        t_elapsed_from_last_step = t_now - self.last_time
        real_xyz = self._std_vector(reward_components, self.rew_min, self.rew_max)
        target_xyz = self._predict_xyz(t_elapsed_from_start)
        next_target_xyz = self._predict_xyz(t_elapsed_from_start + self.dt)
        obs = np.hstack([real_xyz, target_xyz, self.last_reward_components, next_target_xyz])
        if self.use_signals:
            obs = np.append(obs, self._std_vector(np.array(input_output_obs)[self.obs_index], self.obs_input_output_min, self.obs_input_output_max))
        reward = self._calculate_reward(real_xyz, target_xyz, t_elapsed_from_last_step)
        self.last_reward_components = np.array(real_xyz)
        self.last_target_components = np.array(target_xyz)
        self.last_time = t_now
        done = False
        return obs, reward, done, {'r': reward}

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        obs, reward_components, t_state = get_state(self.server, self.id)
        return obs, reward_components

    def _set_action(self, action):
        action_list = []
        for ai, a in enumerate(action):
            if ai in self.binary_signals:
                action_list.append(self.act_min[ai] + (self.act_max[ai] - self.act_min[ai]) * np.round((a - self.act_min[ai]) / (self.act_max[ai] - self.act_min[ai])))
            else:
                action_list.append(a)
        self.actions.append(action_list)
        set_action(self.server, self.id, action_list)

    def _std_vector(self, vector, xmin, xmax, eps=1e-10):
        vector = (vector - xmin) / (xmax - xmin + eps)
        return vector

    def _orig_vector(self, vector, xmin, xmax):
        vector = vector * (xmax - xmin) + xmin
        return vector

    def _predict_xyz(self, t):
        xyz_interp = np.hstack([
            np.interp(t, self.rew_t, self.rew_xyz[:, 0]),
            np.interp(t, self.rew_t, self.rew_xyz[:, 1]),
            np.interp(t, self.rew_t, self.rew_xyz[:, 2])
        ])
        return xyz_interp

    def _calculate_reward(self, real_xyz, target_xyz, t_elapsed_from_last_step, reward_x_coef=0.5, reward_dxdt_coef=0.0):
        dxyzdt = (real_xyz - self.last_reward_components) / t_elapsed_from_last_step
        target_dxyzdt = (target_xyz - self.last_target_components) / t_elapsed_from_last_step
        reward_x = np.linalg.norm(real_xyz - target_xyz) / self.dx_max
        reward_dxdt = np.linalg.norm(dxyzdt - target_dxyzdt) / self.dxdt_max
        score = - reward_x_coef * reward_x - reward_dxdt_coef * reward_dxdt
        return score


