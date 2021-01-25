import gym, requests
import numpy as np

from time import sleep, time
from common.data_utils import load_signals, load_trajectory
from common.server_utils import post_signals, get_state, set_action

class MantsinenBasic(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, mvs, signal_csv, identical_input_signals, trajectory_csv, server_url, frequency, input_scales={0: 700}):

        # init params

        self.mvs = mvs
        self.id = None
        self.server = server_url
        self.act_freq = frequency

        # load signals and their limits

        self.signals, mins, maxs = {}, {}, {}
        for key in signal_csv.keys():
            values, xmin, xmax = load_signals(signal_csv[key])
            self.signals[key] = values
            mins[key] = xmin
            maxs[key] = xmax

        # deal with identical signals

        signals_to_remove = []
        signals_to_remove_parents = []
        for i,signal in enumerate(self.signals['input']):
            if signal in identical_input_signals.keys():
                for signal_to_remove in identical_input_signals[signal]:
                    if signal_to_remove in self.signals['input'] and signal_to_remove not in signals_to_remove:
                        signals_to_remove.append(signal_to_remove)
                        signals_to_remove_parents.append(i)
        self.input_index = []
        input_reverse_index = []
        signals_to_stay = []
        for i,signal in enumerate(self.signals['input']):
            if signal in signals_to_remove:
                idx = signals_to_remove.index(signal)
                input_reverse_index.append(signals_to_remove_parents[idx])
            else:
                self.input_index.append(i)
                input_reverse_index.append(i)
                signals_to_stay.append(i)
        self.act_index = []
        for i in input_reverse_index:
            self.act_index.append(signals_to_stay.index(i))
        self.obs_index = np.hstack([self.input_index, np.arange(len(self.signals['input']), len(self.signals['input']) + len(self.signals['output']) + len(self.signals['reward']))])
        self.act_index = np.array(self.act_index)

        # action scales

        self.act_scales = np.ones_like(self.act_index)
        for key in input_scales:
            self.act_scales[key] = input_scales[key]

        # dimensions

        obs_dim = len(self.obs_index) + len(self.signals['reward'])
        act_dim = len(self.input_index)
        rew_dim = len(self.signals['reward'])

        # load trajectory

        self.rew_t, self.rew_xyz = load_trajectory(trajectory_csv)
        self.rew_xyz = self.rew_xyz[:, :rew_dim]
        dt = np.linalg.norm(self.rew_t[:, None, None] - self.rew_t[None, :, None], axis=-1)
        dx = np.linalg.norm(self.rew_xyz[:, None, :] - self.rew_xyz[None, :, :], axis=-1)
        self.dx_max = np.max(dx)
        self.dxdt_max = np.nanmax(dx / dt)

        # set standardization vectors

        if len(mins['reward']) > 0 and len(maxs['reward']) > 0:
            assert len(mins['reward']) == len(maxs['reward'])
            self.rew_min, self.rew_max = np.array(mins['reward']), np.array(maxs['reward'])
        else:
            self.rew_min, self.rew_max = np.zeros(rew_dim), np.ones(rew_dim)
        if len(mins['input']) > 0 and len(maxs['input']) > 0:
            assert len(mins['input']) == len(maxs['input'])
            self.act_min, self.act_max = np.array(mins['input'])[self.input_index], np.array(maxs['input'])[self.input_index]
        else:
            self.act_min, self.act_max = np.zeros(act_dim), np.ones(act_dim)
        if len(mins['input']) > 0 and len(maxs['input']) > 0 and len(mins['output']) > 0 and len(maxs['output']) > 0 and len(mins['reward']) > 0 and len(maxs['reward']) > 0:
            assert len(mins['output']) == len(maxs['output'])
            self.obs_min = np.hstack([np.hstack([mins['input'], mins['output'], mins['reward'], ])[self.obs_index], np.zeros(len(mins['reward']))])
            self.obs_max = np.hstack([np.hstack([maxs['input'], maxs['output'], maxs['reward']])[self.obs_index], np.ones(len(mins['reward']))])
        else:
            self.obs_min, self.obs_max = np.zeros(obs_dim), np.ones(obs_dim)

        # set spaces

        self.observation_space = gym.spaces.Box(low=self.obs_min, high=self.obs_max, shape=(obs_dim,), dtype=np.float)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,), dtype=np.float)

    def reset(self):
        post_signals(self.server, self.id, self.signals)
        obs, reward_components = [], []
        while len(obs) == 0 or len(reward_components) == 0:
            sleep(self.act_freq)
            obs, reward_components = self._get_state()

        obs = np.hstack([np.array(obs)[self.obs_index], self.rew_xyz[0, :]])
        obs = self._std_vector(obs, self.obs_min, self.obs_max)
        self.last_reward_components = self._std_vector(reward_components, self.rew_min, self.rew_max)
        t_now = time()
        self.start_time = t_now
        self.last_time = t_now
        return obs

    def step(self, action):
        action = self._std_vector(action, self.action_space.low, self.action_space.high)
        action = self._orig_vector(action, self.act_min, self.act_max)
        self._set_action(action[self.act_index])
        sleep(self.act_freq)
        obs, reward_components = self._get_state()
        reward_components = self._std_vector(reward_components, self.rew_min, self.rew_max)
        reward, xyz_interp = self._calculate_reward(reward_components)
        obs = np.hstack([np.array(obs)[self.obs_index], xyz_interp])
        obs = self._std_vector(obs, self.obs_min, self.obs_max)
        self.last_reward_components = np.array(reward_components)
        self.last_time = time()
        done = False
        return obs, reward, done, {'r': reward}

    def render(self, mode='human', close=False):
        pass

    def _get_state(self):
        obs, reward_components, t_state = get_state(self.server, self.id)
        return obs, reward_components

    def _set_action(self, action):
        action = [np.round(item) * scale for item, scale in zip(action, self.act_scales)]
        set_action(self.server, self.id, action)

    def _std_vector(self, vector, xmin, xmax, eps=1e-10):
        vector = (vector - xmin) / (xmax - xmin + eps)
        return vector

    def _orig_vector(self, vector, xmin, xmax):
        vector = vector * (xmax - xmin) + xmin
        return vector

    def _calculate_reward(self, xyz, reward_x_coef=0.5, reward_dx_coef=0.5):
        t_now = time()
        t_elapsed_from_the_last_step = t_now - self.last_time
        t_elapsed_from_the_start = t_now - self.start_time
        xyz_interp = np.hstack([
            np.interp(t_elapsed_from_the_start, self.rew_t, self.rew_xyz[:, 0]),
            np.interp(t_elapsed_from_the_start, self.rew_t, self.rew_xyz[:, 1]),
            np.interp(t_elapsed_from_the_start, self.rew_t, self.rew_xyz[:, 2])
        ])
        xyz_interp = self._std_vector(xyz_interp, self.rew_min, self.rew_max)
        xyz_last_interp = np.hstack([
            np.interp(t_elapsed_from_the_start - t_elapsed_from_the_last_step, self.rew_t, self.rew_xyz[:, 0]),
            np.interp(t_elapsed_from_the_start - t_elapsed_from_the_last_step, self.rew_t, self.rew_xyz[:, 1]),
            np.interp(t_elapsed_from_the_start - t_elapsed_from_the_last_step, self.rew_t, self.rew_xyz[:, 2])
        ])
        xyz_last_interp = self._std_vector(xyz_last_interp, self.rew_min, self.rew_max)
        dxyz_interp = (xyz_interp - xyz_last_interp) / t_elapsed_from_the_last_step
        dxyz = (xyz - self.last_reward_components) / t_elapsed_from_the_last_step
        reward_x = np.linalg.norm(xyz - xyz_interp) # / self.dx_max
        reward_dx = np.linalg.norm(dxyz - dxyz_interp) # / self.dxdt_max
        score = - reward_x_coef *  reward_x - reward_dx_coef * reward_dx
        return score, xyz_interp


