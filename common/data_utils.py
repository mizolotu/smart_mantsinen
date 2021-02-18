import pandas, os
import numpy as np
import os.path as osp

def load_signals(dpath, key, postfix='.csv'):
    p = pandas.read_csv(osp.join(dpath, '{0}{1}'.format(key, postfix)), header=None)
    v = p.values
    result = []
    for i in range(v.shape[1]):
        result.append(v[:, i].tolist())
    return result

def update_signals(dpath, key, signals, mins, maxs, postfix='.csv'):
    signals = np.array(signals)
    arr = np.hstack([signals.reshape(-1, 1), mins.reshape(-1, 1), maxs.reshape(-1, 1)])
    pandas.DataFrame(arr).to_csv(osp.join(dpath, '{0}{1}'.format(key, postfix)), index=False, header=None)

def save_trajectory(signal_caps, timestamps, rewards, states, output, with_header=True):
    timestamps = np.array(timestamps)
    rewards = np.array(rewards)
    states = np.array(states)
    trajectory = np.hstack([timestamps[:, None] - timestamps[0], rewards[:, :], states[:, :-3]])
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

def isfloat(value):
    try:
        float(value)
        result = True
    except:
        result = False
    return result

def isempty(value):
    if value is None or value == 'None':
        result = True
    else:
        result = False
    return  result

def parse_conditional_signals(signals, input_signals):
    assert len(signals) == 3
    names, values, conditions = signals
    conditional_signals = []
    for name, value, condition in zip(names, values, conditions):
        if isfloat(value) and isempty(condition):
            conditional_signals.append({'type': 'unconditional', 'name': name, 'value': float(value)})
        elif isfloat(value) and not isempty(condition):
            conditional_signals.append({'type': 'conditional', 'name': name, 'value': float(value), 'condition': condition})
        elif not isfloat(value) and value in input_signals:
            conditional_signals.append({'type': 'identical', 'name': name, 'value': value})
    return conditional_signals

def is_moving(conditional_signals, conditional_values, t):
    values_to_check = []
    if len(conditional_values) > 0 and t is not None:
        for signal, value in zip(conditional_signals, conditional_values):
            if signal['type'] == 'unconditional':
                values_to_check.append(value)
    return np.any(np.array(values_to_check) > 0)

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

def prepare_trajectories(signal_dir, trajectory_files, use_signals=False, action_scale=1, npoints=128, lookback=4):

    # load signals and their limits

    signals, mins, maxs = {}, {}, {}
    for key in ['input', 'output', 'reward']:
        values, xmin, xmax = load_signals(signal_dir, key)
        signals[key] = values
        mins[key] = xmin
        maxs[key] = xmax

    # set standardization vectors

    rew_min, rew_max = np.array(mins['reward']), np.array(maxs['reward'])
    act_min, act_max = np.array(mins['input']), np.array(maxs['input'])
    obs_input_output_min = np.hstack([np.array(mins['input']), mins['output']])
    obs_input_output_max = np.hstack([np.array(maxs['input']), maxs['output']])
    eps = 1e-10

    # loop through trajectories

    trajectories = []
    waypoints = []

    for fpath in trajectory_files:

        # load trajectories

        rew_t, inputs, outputs, rewards = load_trajectory(fpath, signals)

        # calculate way points

        n = len(rew_t)
        assert n >= (npoints * lookback)
        xp = np.arange(n)
        x = np.arange(1, npoints + 1) * n / npoints
        wpoints = np.zeros((npoints + 1, 3))
        for i in range(3):
            wpoints[:-1, i] = np.interp(x, xp, rewards[:, i])
        wpoints[-1, :] = rewards[-1, :]  # add the last waypoint one more time
        waypoints.append(wpoints)

        # standardize data

        rews = (rewards - rew_min[None, :]) / (rew_max[None, :] - rew_min[None, :] + eps)
        ios = (np.hstack([inputs, outputs]) - obs_input_output_min[None, :]) / (obs_input_output_max[None, :] - obs_input_output_min[None, :] + eps)
        ios_with_lookback = np.hstack([ios[k : n - lookback + k + 1, :] for k in range(lookback)])
        inps = np.array(inputs)  # -inf..inf
        inps = (inps - act_min[None, :]) / (act_max[None, :] - act_min[None, :] + 1e-10)  # 0..1
        inps = action_scale * (2 * inps - 1)  # -scale..scale
        inps = inps[lookback - 1 :, :]

        # loop through way points

        for i in range(npoints):
            wp = (wpoints[i, :] - rew_min) / (rew_max - rew_min + eps)
            wp_next = (wpoints[i + 1, :] - rew_min) / (rew_max - rew_min + eps)
            idx = np.where(xp < x[i])[0]
            n = len(idx)

            # observing tool corrdinates

            rp = rews[:n, :]
            from_rp_to_wp = wp - rp
            from_rp_to_wp_next = wp_next - rp
            from_rp_to_wp_with_lookback = np.hstack([from_rp_to_wp[k : n - lookback + k + 1, :] for k in range(lookback)])
            from_rp_to_wp_next_with_lookback = np.hstack([from_rp_to_wp_next[k : n - lookback + k + 1, :] for k in range(lookback)])
            traj = np.hstack([from_rp_to_wp_with_lookback, from_rp_to_wp_next_with_lookback])

            # add signal values

            if use_signals:
                traj = np.append(traj, ios_with_lookback[: n - lookback + 1, :], axis=1)

            # add actions

            traj = np.append(traj, inps[:n - lookback + 1, :], axis=1)

            # add to the list

            trajectories.append(traj)

    return np.vstack(trajectories), waypoints