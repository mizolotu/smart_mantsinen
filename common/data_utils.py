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

def prepare_trajectories(signal_dir, trajectory_files, use_signals=False, action_scale=1, lookback=4, wp_dist=0.1):

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

    # for bonus

    min_bonus = -np.inf

    for fpath in trajectory_files:

        max_d_to_second_nearest_wp_std = -np.inf

        # load trajectories

        rew_t, inputs, outputs, rewards = load_trajectory(fpath, signals)

        # calculate way points

        n = len(rew_t)
        wpoints = []
        x = []
        xyz = rewards[0, :]
        for i in range(n):
            d = np.linalg.norm(xyz - rewards[i, :])
            if d > wp_dist:
                xyz = rewards[i, :]
                wpoints.append(xyz)
                x.append(i)
        wpoints = np.vstack(wpoints)
        waypoints.append(wpoints)

        # standardize data

        wps_std = (wpoints - rew_min[None, :]) / (rew_max[None, :] - rew_min[None, :] + eps)
        rews_std = (rewards - rew_min[None, :]) / (rew_max[None, :] - rew_min[None, :] + eps)
        rews_std_with_lookback = np.hstack([rews_std[k: n - lookback + k + 1, :] for k in range(lookback)])
        ios_std = (np.hstack([inputs, outputs]) - obs_input_output_min[None, :]) / (obs_input_output_max[None, :] - obs_input_output_min[None, :] + eps)
        ios_std_with_lookback = np.hstack([ios_std[k : n - lookback + k + 1, :] for k in range(lookback)])
        inps = np.array(inputs)  # -inf..inf
        inps = (inps - act_min[None, :]) / (act_max[None, :] - act_min[None, :] + 1e-10)  # 0..1
        inps = action_scale * (2 * inps - 1)  # -scale..scale
        inps = inps[lookback - 1 :, :]

        # loop through trajectory points

        npoints = wpoints.shape[0]
        wp_idx = np.zeros(npoints)
        for i in range(lookback - 1, n):
            dist_to_wps = np.linalg.norm(wpoints - rewards[i, :], axis=1)
            idx_sorted = np.argsort(dist_to_wps)
            wp_nearest_idx = idx_sorted[0]
            wp_idx[wp_nearest_idx] += 1
            if wp_nearest_idx < (npoints - 1):
                wp_next_idx = wp_nearest_idx + 1
            else:
                wp_next_idx = wp_nearest_idx
            #idx_unvisited = np.where(wp_idx == 0)[0]
            #wp_unvisited_nearest_idx = np.argmin(dist_to_wps[idx_unvisited])
            wp = wps_std[wp_next_idx, :]

            # for bonus

            wp_second_nearest_idx = idx_sorted[1]
            d_to_second_nearest_wp_std = np.linalg.norm(rews_std[i, :] - wps_std[wp_second_nearest_idx, :])
            if d_to_second_nearest_wp_std > max_d_to_second_nearest_wp_std:
                max_d_to_second_nearest_wp_std = d_to_second_nearest_wp_std

            # creating obs

            idx = i - lookback + 1
            rp_with_loopback = rews_std_with_lookback[idx, :].reshape(lookback, 3)
            from_rp_to_wp_with_lookback = wp - rp_with_loopback
            #from_rp_to_wp_next_with_lookback = wp_next - rp_with_loopback
            #from_rp_to_wp_with_lookback = np.hstack([from_rp_to_wp[k : n - lookback + k + 1, :] for k in range(lookback)])
            #from_rp_to_wp_next_with_lookback = np.hstack([from_rp_to_wp_next[k : n - lookback + k + 1, :] for k in range(lookback)])
            #traj = np.hstack([from_rp_to_wp_with_lookback.reshape(1, -1), from_rp_to_wp_next_with_lookback.reshape(1, -1)])[0]
            traj = from_rp_to_wp_with_lookback.reshape(1, -1)[0]

            # add signal values

            if use_signals:
                traj = np.append(traj, ios_std_with_lookback[idx, :])

            # add actions

            traj = np.append(traj, inps[idx, :])

            # add to the list

            trajectories.append(traj)

        # for bonus

        if (npoints * max_d_to_second_nearest_wp_std) > min_bonus:
            min_bonus = npoints * max_d_to_second_nearest_wp_std

    return np.vstack(trajectories), waypoints, min_bonus