import pandas
import numpy as np

def load_signals(fpath):
    xmin = []
    xmax = []
    p = pandas.read_csv(fpath, header=None)
    v = p.values
    signals = v[:, 0].tolist()
    if v.shape[1] > 1:
        xmin = v[:, 1].tolist()
        xmax = v[:, 2].tolist()
    return signals, xmin, xmax

def update_signals(fpath, signals, mins, maxs):
    signals = np.array(signals)
    arr = np.hstack([signals.reshape(-1, 1), mins.reshape(-1, 1), maxs.reshape(-1, 1)])
    pandas.DataFrame(arr).to_csv(fpath, index=False, header=None)

def save_trajectory(signal_caps, timestamps, rewards, states, stay_t_start, output, with_header=True):
    timestamps = np.array(timestamps)
    rewards = np.array(rewards)
    states = np.array(states)
    idx = np.where(timestamps < stay_t_start)[0]
    trajectory = np.hstack([timestamps[idx, None] - timestamps[idx[0]], rewards[idx, :], states[idx, :-3]])
    if with_header:
        header = np.hstack(['Timestamp', signal_caps['reward'], signal_caps['input'], signal_caps['output']])
        trajectory = np.vstack([header, trajectory])
    pandas.DataFrame(trajectory).to_csv(output, index=False, header=False)

def load_trajectory(fpath, signals):
    p = pandas.read_csv(fpath, header=0)
    t = p['Timestamp'].values
    inputs = p[signals['input']].values
    outputs = p[signals['output']].values
    rewards = p[signals['reward']].values
    return t, inputs, outputs, rewards

def adjust_indexes(signals, identical_input_signals):
    signals_to_remove = []
    signals_to_remove_parents = []
    for i, signal in enumerate(signals['input']):
        if signal in identical_input_signals.keys():
            for signal_to_remove in identical_input_signals[signal]:
                if signal_to_remove in signals['input'] and signal_to_remove not in signals_to_remove:
                    signals_to_remove.append(signal_to_remove)
                    signals_to_remove_parents.append(i)
    obs_input_index = []
    input_reverse_index = []
    signals_to_stay = []
    for i, signal in enumerate(signals['input']):
        if signal in signals_to_remove:
            idx = signals_to_remove.index(signal)
            input_reverse_index.append(signals_to_remove_parents[idx])
        else:
            obs_input_index.append(i)
            input_reverse_index.append(i)
            signals_to_stay.append(i)
    act_index = []
    for i in input_reverse_index:
        act_index.append(signals_to_stay.index(i))
    obs_input_index = np.array(obs_input_index)
    act_index = np.array(act_index)
    return obs_input_index, act_index

def prepare_trajectories(signal_csv, identical_input_signals, trajectory_files, dt, use_signals=False, interp=False):

    # load signals and their limits

    signals, mins, maxs = {}, {}, {}
    for key in signal_csv.keys():
        values, xmin, xmax = load_signals(signal_csv[key])
        signals[key] = values
        mins[key] = xmin
        maxs[key] = xmax

    # deal with identical signals

    obs_input_idx, act_idx = adjust_indexes(signals, identical_input_signals)

    # set standardization vectors

    rew_min, rew_max = np.array(mins['reward']), np.array(maxs['reward'])
    act_min, act_max = np.array(mins['input'])[obs_input_idx], np.array(maxs['input'])[obs_input_idx]
    obs_input_output_min = np.hstack([np.array(mins['input'])[obs_input_idx], mins['output']])
    obs_input_output_max = np.hstack([np.array(maxs['input'])[obs_input_idx], maxs['output']])
    eps = 1e-10

    # collect trajectories

    trajectories = []
    for fpath in trajectory_files:

        # load trajectories

        rew_t, inputs, outputs, rewards = load_trajectory(fpath, signals)

        # add observations

        rewards_current = (rewards[1:] - rew_min[None, :]) / (rew_max[None, :] - rew_min[None, :] + eps)
        rewards_previous = (rewards[:-1, :] - rew_min[None, :]) / (rew_max[None, :] - rew_min[None, :] + eps)
        rewards_next = np.zeros_like(rewards_current)
        for i in range(rewards_current.shape[0]):
            for j in range(rewards_current.shape[1]):
                rewards_next[i, j] = np.interp(rew_t[i] + dt, rew_t[1:], rewards_current[:, j])
        rewards_target = np.zeros_like(rewards_current)
        for i in range(rewards_current.shape[0]):
            for j in range(rewards_current.shape[1]):
                rewards_next[i, j] = np.interp(rew_t[i], rew_t[1:], rewards_current[:, j])
        inputs = inputs[1:, :]
        outputs = outputs[1:, :]
        traj = np.hstack([rewards_current, rewards_target, rewards_previous, rewards_next])
        if use_signals:
            io = np.hstack([inputs[:, obs_input_idx], outputs[:, :]])
            io = (io - obs_input_output_min[None, :]) / (obs_input_output_max[None, :] - obs_input_output_min[None, :] + 1e-10)
            traj = np.append(traj, io, axis=1)

        # add actions

        i = inputs[:, obs_input_idx]  # -inf..inf
        i = (i - act_min[None, :]) / (act_max[None, :] - act_min[None, :] + 1e-10)  # 0..1
        i = 2 * i -1 # -1..1
        traj = np.append(traj, i, axis=1)

        # add timestamps

        traj = np.append(traj, rew_t[1:, None], axis=1)

        trajectories.append(traj)

    return trajectories