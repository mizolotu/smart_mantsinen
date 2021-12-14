import gym
import numpy as np

from time import sleep
from common.data_utils import load_signals, parse_conditional_signals
from common.server_utils import post_signals, get_state, post_action
from collections import deque

class MantsinenBasic(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_idx, mvs, env_dir, signal_dir, server_url, waypoints, nsteps, lookback,
                 use_inputs, use_outputs, scale, wp_size, tstep, bonus):

        # init params

        self.mvs = mvs
        self.dir = env_dir
        self.env_idx = env_idx
        self.id = None
        self.server = server_url
        if type(waypoints).__name__ == 'list':
            self.waypoints_list = waypoints
            self.waypoints = waypoints[0]
        else:
            self.waypoints_list = [waypoints]
            self.waypoints = waypoints
        self.nsteps = nsteps
        self.lookback = lookback
        self.use_inputs = use_inputs
        self.use_outputs = use_outputs
        self.tstep = tstep
        self.bonus = bonus
        self.wp_size = wp_size
        self.wps_completed = np.zeros(self.waypoints.shape[0])

        self.rp_to_wp_buff = deque(maxlen=lookback)
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

        npoints = 1
        ndists = 1
        rew_dim = len(self.signals['reward'])
        obs_dim = (rew_dim * npoints + ndists) * lookback
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
            obs_dim += len(self.obs_input_index)
        if use_outputs:
            obs_dim += len(self.obs_output_index)
        act_dim = len(self.signals['input'])
        self.act_min, self.act_max = np.array(mins['input']), np.array(maxs['input'])

        # times and counts

        self.step_simulation_time = None
        self.last_simulation_time = None
        self.total_step_count = 0

        # set spaces

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float)
        self.action_space = gym.spaces.Box(low=-scale, high=scale, shape=(act_dim,), dtype=np.float)

    def reset(self, init_sleep=0.1):

        # reset times and counts

        self.last_simulation_time = None
        self.step_simulation_time = None
        self.step_count = 0
        self.cum_dist_to_last = 0
        self.reward = 0

        # post signals

        post_signals(self.server, self.id, self.signals)

        # wait until the solver starts

        input_output_obs, reward_components = [], []
        while len(input_output_obs) == 0 or len(reward_components) == 0:
            sleep(init_sleep)
            input_output_obs, reward_components, last_state_time, crashed = self._get_state()

        # clear and fill observation buffers

        self.rp_to_wp_buff.clear()
        self.i_buff.clear()
        self.o_buff.clear()
        input_output_obs, reward_components, last_state_time, crashed = self._get_state()
        xyz = np.array(reward_components)
        i = np.array(input_output_obs)[self.obs_input_index]
        i_std = self._std_vector(i, self.obs_input_min, self.obs_input_max)
        self.i_buff.append(i_std)
        o = np.array(input_output_obs)[self.obs_output_index]
        o_std = self._std_vector(o, self.obs_output_min, self.obs_output_max)
        self.o_buff.append(o_std)

        # update simulation times

        self.step_simulation_time = last_state_time
        self.last_simulation_time = last_state_time

        # select waypoints

        idx = np.random.choice(len(self.waypoints_list))
        self.waypoints = self.waypoints_list[idx]
        self.wps_completed = np.zeros(self.waypoints.shape[0])

        # calculate obs

        wp_nearst, wp_nearest_not_completed = self._calculate_relations_to_wps(xyz)
        obs = self._calculate_obs(xyz, wp_nearest_not_completed)

        return obs

    def step(self, action):
        print(action)

        # set action

        action = self._std_vector(action, self.action_space.low, self.action_space.high)
        action = self._orig_vector(action, self.act_min, self.act_max)
        self._set_action(action)
        self.step_count += 1
        self.total_step_count +=1

        # get state and update simulation time

        #print(self.env_idx, self.step_count, self.total_step_count % self.nsteps, 'before get_state')
        input_output_obs, reward_components, self.last_simulation_time, crashed = self._get_state()
        #print(self.env_idx, self.step_count, self.total_step_count % self.nsteps, 'after get_state')
        new_state_simulation_time_max = self.step_simulation_time + self.tstep
        if self.last_simulation_time > new_state_simulation_time_max:
            self.step_simulation_time = self.last_simulation_time
        else:
            self.step_simulation_time += self.tstep

        # calculate reward

        xyz = np.array(reward_components)
        wp_nearst, wp_nearest_not_completed = self._calculate_relations_to_wps(xyz)
        reward, done, info, switch_wp = self._calculate_reward(xyz)
        self.reward += reward

        # crashed?

        if crashed:
            print(id, 'Crashed :(')
            done = True

        # switch wp

        if switch_wp:
            self.rp_to_wp_buff.clear()
            self.i_buff.clear()
            self.o_buff.clear()

        # calculate new obs

        i = np.array(input_output_obs)[self.obs_input_index]
        i_std = self._std_vector(i, self.obs_input_min, self.obs_input_max)
        self.i_buff.append(i_std)
        o = np.array(input_output_obs)[self.obs_output_index]
        o_std = self._std_vector(o, self.obs_output_min, self.obs_output_max)
        self.o_buff.append(o_std)
        obs = self._calculate_obs(xyz, wp_nearest_not_completed)

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        obs, reward_components, conditional_components, t_state_real, t_state_simulation, crashed = get_state(
            self.server, self.id, self.step_simulation_time, self.step_count == self.nsteps, self.step_count
        )
        return obs, reward_components, t_state_simulation, crashed

    def _std_vector(self, vector, xmin, xmax, eps=1e-10):
        vector = (vector - xmin) / (xmax - xmin + eps)
        return vector

    def _orig_vector(self, vector, xmin, xmax):
        vector = vector * (xmax - xmin) + xmin
        return vector

    def _calculate_relations_to_wps(self, xyz):
        wps_not_completed_idx = np.where(self.wps_completed == 0)[0]
        if len(wps_not_completed_idx) > 0:
            dist_to_wps = np.linalg.norm(self.waypoints - xyz, axis=1)
            idx_min_all = np.argmin(dist_to_wps)
            idx_min_not_completed = np.argmin(dist_to_wps[wps_not_completed_idx])
            wp_nearest = self.waypoints[idx_min_all, :]
            wp_nearest_not_completed = self.waypoints[wps_not_completed_idx[idx_min_not_completed], :]
        else:
            wp_nearest_not_completed = self.waypoints[-1, :]
        return wp_nearest, wp_nearest_not_completed

    def _calculate_obs(self, xyz, wp_nearest_not_completed):

        from_rp_to_wp_first = self.waypoints[0, :] - xyz
        from_rp_to_wp_first_norm = np.linalg.norm(from_rp_to_wp_first)
        from_rp_to_wp_first /= (from_rp_to_wp_first_norm + 1e-10)
        from_rp_to_wp_first_norm_std = from_rp_to_wp_first_norm / self.d_max

        from_rp_to_wp_nearest = wp_nearest_not_completed - xyz
        from_rp_to_wp_nearest_norm = np.linalg.norm(from_rp_to_wp_nearest)
        from_rp_to_wp_nearest /= (from_rp_to_wp_nearest_norm + 1e-10)
        from_rp_to_wp_nearest_norm_std = from_rp_to_wp_nearest_norm / self.d_max

        from_rp_to_wp_last = self.waypoints[-1, :] - xyz
        from_rp_to_wp_last_norm = np.linalg.norm(from_rp_to_wp_last)
        from_rp_to_wp_last /= (from_rp_to_wp_last_norm + 1e-10)
        from_rp_to_wp_last_norm_std = from_rp_to_wp_last_norm / self.d_max

        rp_to_wp = np.hstack([
            #from_rp_to_wp_first,
            #from_rp_to_wp_first_norm_std,
            from_rp_to_wp_nearest,
            from_rp_to_wp_nearest_norm_std,
            #from_rp_to_wp_last,
            #from_rp_to_wp_last_norm_std
        ])

        self.rp_to_wp_buff.append(rp_to_wp)
        obs = np.vstack(self.rp_to_wp_buff)

        # pad with zeros

        zeros = np.zeros((self.lookback - obs.shape[0], obs.shape[1]))
        obs = np.vstack([obs, zeros])
        obs = obs.reshape(1, -1).flatten()

        if self.use_inputs:
            i = self.i_buff[-1]
            obs = np.hstack([obs, i])
        if self.use_outputs:
            o = self.o_buff[-1]
            obs = np.hstack([obs, o])

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
        dists_to_wps = np.linalg.norm(xyz - self.waypoints, axis=1)
        wp_nearest_idx = np.argmin(dists_to_wps)
        dist_to_nearest_wp = dists_to_wps[wp_nearest_idx]
        dist_to_nearest_wp_std = dist_to_nearest_wp / self.d_max
        dist_to_last = np.linalg.norm(xyz - self.waypoints[-1, :])
        dist_to_last_std = dist_to_last / self.d_max
        if dist_to_nearest_wp <= self.wp_size and self.wps_completed[wp_nearest_idx] == 0:
            self.wps_completed[wp_nearest_idx] = 1
            score = np.clip(1 - dist_to_nearest_wp_std - dist_to_last_std, 0, 1) + self.bonus
            done = False
            switch_wp = True
        else:
            score = np.clip(1 - dist_to_nearest_wp_std - dist_to_last_std, 0, 1)
            done = False
            switch_wp = False
        info = {'rc1': dist_to_nearest_wp_std, 'rc2': dist_to_last_std, 'rc3': 0}
        return score, done, info, switch_wp