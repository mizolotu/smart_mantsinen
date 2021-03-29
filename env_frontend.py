import gym
import numpy as np

from time import sleep, time
from common.data_utils import load_signals, parse_conditional_signals
from common.server_utils import post_signals, get_state, post_action
from collections import deque

class MantsinenBasic(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, mvs, signal_dir, server_url, waypoints, lookback, use_inputs, use_outputs, scale, tstep):

        # init params

        self.mvs = mvs
        self.id = None
        self.server = server_url
        self.waypoints = waypoints
        self.lookback = lookback
        self.use_inputs = use_inputs
        self.use_outputs = use_outputs
        self.tstep = tstep
        self.xyz_buff = deque(maxlen=lookback)
        self.i_buff = deque(maxlen=lookback)
        self.o_buff = deque(maxlen=lookback)

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

        self.obs_input_index = np.arange(len(self.signals['input']))
        self.obs_output_index = np.arange(len(self.signals['input']), len(self.signals['input']) + len(self.signals['output']))
        self.obs_index = np.arange(len(self.signals['input']) + len(self.signals['output']))

        # dimensions and standardization coefficients

        npoints = 5  # number of waypoints participating in each observation calculation
        rew_dim = len(self.signals['reward'])
        obs_dim = lookback * rew_dim * npoints
        self.rew_min = np.array(mins['reward'])
        self.rew_max = np.array(maxs['reward'])
        self.v_min = np.hstack([self.rew_min - self.rew_max] * lookback)
        self.v_max = np.hstack([self.rew_max - self.rew_min] * lookback)
        self.d_max = np.linalg.norm(self.rew_max - self.rew_min)
        self.obs_input_min = np.array(mins['input'])
        self.obs_input_max = np.array(maxs['input'])
        self.obs_output_min = np.array(mins['output'])
        self.obs_output_max = np.array(maxs['output'])
        if use_inputs:
            obs_dim += len(self.obs_input_index) * lookback
        if use_outputs:
            obs_dim += len(self.obs_output_index) * lookback
        act_dim = len(self.signals['input'])
        self.act_min, self.act_max = np.array(mins['input']), np.array(maxs['input'])

        # times and counts

        self.step_simulation_time = None
        self.last_simulation_time = None
        self.step_count = 0

        # set spaces

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float)
        self.action_space = gym.spaces.Box(low=-scale, high=scale, shape=(act_dim,), dtype=np.float)

    def reset(self, init_sleep=0.1):

        # reset times and counts

        self.last_simulation_time = None
        self.step_simulation_time = None
        self.step_count = 0

        # post signals

        post_signals(self.server, self.id, self.signals)

        # wait until the solver starts

        input_output_obs, reward_components = [], []
        while len(input_output_obs) == 0 or len(reward_components) == 0:
            sleep(init_sleep)
            input_output_obs, reward_components, last_state_time = self._get_state()

        # clear and fill observation buffers

        self.xyz_buff.clear()
        self.i_buff.clear()
        self.o_buff.clear()
        while len(self.xyz_buff) < self.lookback or len(self.i_buff) < self.lookback or len(self.o_buff) < self.lookback:
            input_output_obs, reward_components, last_state_time = self._get_state()
            xyz = np.array(reward_components)
            self.xyz_buff.append(xyz)
            i = np.array(input_output_obs)[self.obs_input_index]
            i_std = self._std_vector(i, self.obs_input_min, self.obs_input_max)
            self.i_buff.append(i_std)
            o = np.array(input_output_obs)[self.obs_output_index]
            o_std = self._std_vector(o, self.obs_output_min, self.obs_output_max)
            self.o_buff.append(o_std)

        # update simulation times

        self.step_simulation_time = last_state_time
        self.last_simulation_time = last_state_time

        # calculate obs

        wp_nearst, wp_before_nearest, wp_after_nearest = self._calculate_relations_to_wps(xyz)
        obs = self._calculate_obs(wp_nearst, wp_before_nearest, wp_after_nearest)

        return obs

    def step(self, action):

        # set action

        action = self._std_vector(action, self.action_space.low, self.action_space.high)
        action = self._orig_vector(action, self.act_min, self.act_max)
        self._set_action(action)
        self.step_count += 1

        # get state and update simulation time

        input_output_obs, reward_components, self.last_simulation_time = self._get_state()
        new_state_simulation_time_max = self.step_simulation_time + self.tstep
        if self.last_simulation_time > new_state_simulation_time_max:
            self.step_simulation_time = self.last_simulation_time
        else:
            self.step_simulation_time += self.tstep

        # calculate reward

        xyz = np.array(reward_components)
        wp_nearst, wp_before_nearest, wp_fater_nearest = self._calculate_relations_to_wps(xyz)
        reward, done, info = self._calculate_reward(xyz)

        # calculate new obs

        self.xyz_buff.append(xyz)
        i = np.array(input_output_obs)[self.obs_input_index]
        i_std = self._std_vector(i, self.obs_input_min, self.obs_input_max)
        self.i_buff.append(i_std)
        o = np.array(input_output_obs)[self.obs_output_index]
        o_std = self._std_vector(o, self.obs_output_min, self.obs_output_max)
        self.o_buff.append(o_std)
        obs = self._calculate_obs(wp_nearst, wp_before_nearest, wp_fater_nearest)

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        obs, reward_components, conditional_components, t_state_real, t_state_simulation = get_state(self.server, self.id, self.step_simulation_time)
        return obs, reward_components, t_state_simulation

    def _std_vector(self, vector, xmin, xmax, eps=1e-10):
        vector = (vector - xmin) / (xmax - xmin + eps)
        return vector

    def _orig_vector(self, vector, xmin, xmax):
        vector = vector * (xmax - xmin) + xmin
        return vector

    def _calculate_relations_to_wps(self, xyz):
        dists_to_wps = np.linalg.norm(xyz - self.waypoints, axis=1)
        wp_nearest_idx = np.argmin(dists_to_wps)
        if wp_nearest_idx > 0:
            wp_before_nearest_idx = wp_nearest_idx - 1
        else:
            wp_before_nearest_idx = wp_nearest_idx
        if wp_nearest_idx < len(dists_to_wps) - 1:
            wp_after_nearest_idx = wp_nearest_idx + 1
        else:
            wp_after_nearest_idx = wp_nearest_idx
        wp_nearest = self.waypoints[wp_nearest_idx, :]
        wp_before_nearest = self.waypoints[wp_before_nearest_idx, :]
        wp_after_nearest = self.waypoints[wp_after_nearest_idx, :]
        return wp_nearest, wp_before_nearest, wp_after_nearest

    def _calculate_obs(self, wp_nearest, wp_before_nearest, wp_after_nearest):
        rp = np.vstack(self.xyz_buff)
        from_rp_to_wp_nearest = wp_nearest - rp
        from_rp_to_wp_before_nearest = wp_before_nearest - rp
        from_rp_to_wp_after_nearest = wp_after_nearest - rp
        from_rp_to_wp_first = self.waypoints[0, :] - rp
        from_rp_to_wp_last = self.waypoints[-1, :] - rp
        obs = np.hstack([
            self._std_vector(from_rp_to_wp_first.reshape(1, -1)[0], self.v_min, self.v_max),
            self._std_vector(from_rp_to_wp_before_nearest.reshape(1, -1)[0], self.v_min, self.v_max),
            self._std_vector(from_rp_to_wp_nearest.reshape(1, -1)[0], self.v_min, self.v_max),
            self._std_vector(from_rp_to_wp_after_nearest.reshape(1, -1)[0], self.v_min, self.v_max),
            self._std_vector(from_rp_to_wp_last.reshape(1, -1)[0], self.v_min, self.v_max)
        ])
        if self.use_inputs:
            i = np.vstack(self.i_buff).reshape(1, -1)[0]
            obs = np.append(obs, i)
        if self.use_outputs:
            o = np.vstack(self.o_buff).reshape(1, -1)[0]
            obs = np.append(obs, o)
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
        post_action(self.server, self.id, action.tolist(), conditional, self.tstep)

    def _calculate_reward(self, xyz):
        done = False
        dists_to_wps = np.linalg.norm(xyz - self.waypoints, axis=1)
        idx_sorted = np.argsort(dists_to_wps)
        wp_nearest_idx = idx_sorted[0]
        dist_to_nearest_wp = dists_to_wps[wp_nearest_idx]
        dist_to_nearest_wp_std = dist_to_nearest_wp / self.d_max
        dist_to_last = np.linalg.norm(xyz - self.waypoints[-1, :])
        dist_to_last_std = dist_to_last / self.d_max
        score = - dist_to_nearest_wp_std - dist_to_last_std
        info = {'rc1': -dist_to_nearest_wp_std, 'rc2': -dist_to_last_std, 'rc3': 0}
        return score, done, info