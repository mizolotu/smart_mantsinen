import gym
import numpy as np

from time import sleep, time
from common.data_utils import load_signals, parse_conditional_signals
from common.server_utils import post_signals, get_state, set_action
from collections import deque

class MantsinenBasic(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, mvs, signal_dir, server_url, waypoints, lookback, use_signals, scale, dist_thr, bonus=1):

        # init params

        self.mvs = mvs
        self.id = None
        self.server = server_url
        self.waypoints = waypoints
        self.lookback = lookback
        self.use_signals = use_signals
        self.dist_thr = dist_thr
        self.bonus = bonus
        self.last_action_time = time()
        self.wp_index = 0
        self.xyz_buff = deque(maxlen=lookback)
        self.io_buff = deque(maxlen=lookback)

        # load signals and their limits

        self.signals, mins, maxs = {}, {}, {}
        for key in ['input', 'output', 'reward']:
            values, xmin, xmax = load_signals(signal_dir, key)
            self.signals[key] = values
            mins[key] = xmin
            maxs[key] = xmax

        values = load_signals(signal_dir, 'conditional')
        self.signals['conditional'] = values[0]
        self.conditional_signals = parse_conditional_signals(values, self.signals['input'])

        # observation index

        self.obs_index = np.arange(len(self.signals['input']) + len(self.signals['output']))

        # dimensions and standardization coefficients

        rew_dim = len(self.signals['reward'])
        obs_dim = lookback * 2 * rew_dim
        self.rew_min = np.array(mins['reward'])
        self.rew_max = np.array(maxs['reward'])
        obs_min = np.hstack([self.rew_min - self.rew_max] * 2 * lookback)
        obs_max = np.hstack([self.rew_max - self.rew_min] * 2 * lookback)
        self.obs_input_output_min = np.hstack([np.array(mins['input']), mins['output']])
        self.obs_input_output_max = np.hstack([np.array(maxs['input']), maxs['output']])
        if use_signals:
            obs_dim += len(self.obs_index) * lookback
            obs_min = np.hstack([obs_min, *[self.obs_input_output_min] * lookback])
            obs_max = np.hstack([obs_max, *[self.obs_input_output_max] * lookback])
        act_dim = len(self.signals['input'])
        self.act_min, self.act_max = np.array(mins['input']), np.array(maxs['input'])

        # calculate parameters for reward scaling

        #dt = np.linalg.norm(self.rew_t[:, None, None] - self.rew_t[None, :, None], axis=-1)
        #dx = np.linalg.norm(self.rew_xyz[:, None, :] - self.rew_xyz[None, :, :], axis=-1)
        #self.dx_max = np.max(dx)
        #self.dt_max = np.max(dt)
        #self.dxdt_max = np.nanmax(dx / dt)

        # set spaces

        self.observation_space = gym.spaces.Box(low=obs_min, high=obs_max, shape=(obs_dim,), dtype=np.float)
        self.action_space = gym.spaces.Box(low=-scale, high=scale, shape=(act_dim,), dtype=np.float)

    def reset(self, init_sleep=0.1):

        # post signals

        post_signals(self.server, self.id, self.signals)

        # wait until the solver starts

        input_output_obs, reward_components = [], []
        while len(input_output_obs) == 0 or len(reward_components) == 0:
            sleep(init_sleep)
            input_output_obs, reward_components = self._get_state()

        # fill observation buffers

        while len(self.xyz_buff) < self.lookback or len(self.io_buff) < self.lookback:
            input_output_obs, reward_components = self._get_state()
            xyz = np.array(reward_components)
            xyz_std = self._std_vector(xyz, self.rew_min, self.rew_max)
            self.xyz_buff.append(xyz_std)
            io = np.array(input_output_obs)[self.obs_index]
            io_std = self._std_vector(np.array(io), self.obs_input_output_min, self.obs_input_output_max)
            self.io_buff.append(io_std)

        # set waypoint

        self.wp_index = 0

        # calculate obs

        obs = self._calculate_obs()

        return obs

    def step(self, action):
        action = self._std_vector(action, self.action_space.low, self.action_space.high)
        action = self._orig_vector(action, self.act_min, self.act_max)
        self._set_action(action)
        input_output_obs, reward_components = self._get_state()

        # calculate reward first

        xyz = np.array(reward_components)
        reward, done = self._calculate_reward(xyz)

        # calculate new obs

        xyz_std = self._std_vector(xyz, self.rew_min, self.rew_max)
        io = np.array(input_output_obs)[self.obs_index]
        io_std = self._std_vector(np.array(io), self.obs_input_output_min, self.obs_input_output_max)
        self.xyz_buff.append(xyz_std)
        self.io_buff.append(io_std)
        obs = self._calculate_obs()

        return obs, reward, done, {'r': reward}

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        obs, reward_components, _, t_state = get_state(self.server, self.id)
        return obs, reward_components

    def _std_vector(self, vector, xmin, xmax, eps=1e-10):
        vector = (vector - xmin) / (xmax - xmin + eps)
        return vector

    def _orig_vector(self, vector, xmin, xmax):
        vector = vector * (xmax - xmin) + xmin
        return vector

    def _calculate_obs(self):
        wp = self.waypoints[self.wp_index, :]
        wp_std = self._std_vector(wp, self.rew_min, self.rew_max)
        wp_next = self.waypoints[self.wp_index + 1, :]
        wp_next_std = self._std_vector(wp_next, self.rew_min, self.rew_max)
        rp_std = np.vstack(self.xyz_buff)
        from_rp_to_wp = wp_std - rp_std
        from_rp_to_wp_next = wp_next_std - rp_std
        obs = np.hstack([from_rp_to_wp.reshape(1, -1)[0], from_rp_to_wp_next.reshape(1, -1)[0]])
        if self.use_signals:
            io = np.vstack(self.io_buff).reshape(1, -1)[0]
            obs = np.append(obs, io)
        return obs

    def _set_action(self, action):
        conditional = []
        for signal in self.conditional_signals:
            if signal['type'] == 'unconditional':
                conditional.append(signal['value'])
            elif signal['type'] == 'conditional':
                idx = self.signals['input'].index(signal['condition'])
                if action[idx] > self.act_min[idx]:
                    value = signal['value']
                else:
                    value = 0
                conditional.append(value)
            elif signal['type'] == 'identical':
                idx = self.signals['input'].index(signal['value'])
                conditional.append(action[idx])
        self.last_action_time = set_action(self.server, self.id, action.tolist(), conditional, self.last_action_time)

    def _calculate_reward(self, xyz):
        done = False
        xyz_std = self._std_vector(xyz, self.rew_min, self.rew_max)
        wp = self.waypoints[self.wp_index]
        wp_std = self._std_vector(wp, self.rew_min, self.rew_max)
        d = np.linalg.norm(xyz_std - wp_std)
        score = -d
        if d < self.dist_thr:
            if self.wp_index < (len(self.waypoints) - 1):
                self.wp_index += 1
            else:
                done = True
            score += self.bonus
        return score, done