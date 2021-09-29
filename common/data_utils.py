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

def is_acting(signals, values, values_to_avoid):
    nactions = len(signals['input'])
    result = True
    action = values[:nactions]
    action = [int(item) for item in action]
    for value_to_avoid in values_to_avoid:
        if action == value_to_avoid:
            result = False
            break
    return result

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

def prepare_trajectories(signal_dir, trajectory_files, n_waypoints=16, use_inputs=True, use_outputs=True, action_scale=1, lookback=4, tstep=0.01):

    # load signals and their limits

    signals, mins, maxs = {}, {}, {}
    for key in ['input', 'output', 'reward']:
        values, xmin, xmax = load_signals(signal_dir, key)
        signals[key] = values
        mins[key] = xmin
        maxs[key] = xmax

    # set standardization vectors

    rew_min, rew_max = np.array(mins['reward']), np.array(maxs['reward'])
    v_min = np.hstack([rew_min - rew_max] * lookback)
    v_max = np.hstack([rew_max - rew_min] * lookback)
    d_max = np.linalg.norm(rew_max - rew_min)
    act_min, act_max = np.array(mins['input']), np.array(maxs['input'])
    obs_input_min = np.array(mins['input'])
    obs_input_max = np.array(maxs['input'])
    obs_output_min = np.array(mins['output'])
    obs_output_max = np.array(maxs['output'])
    eps = 1e-10

    # loop through trajectories

    trajectories_tr = []
    trajectories_val = []
    waypoints = []
    n_stay_maxs = []
    last_dists = []
    rews = []

    for ti, fpath in enumerate(trajectory_files):

        # load trajectories

        rew_t, inputs, outputs, rewards = load_trajectory(fpath, signals)
        rews.append(rewards)

        # calculate way points by interpolating over time

        n = len(rew_t)
        duration = rew_t[-1] - rew_t[0]
        waypoint_time_step = duration / (n_waypoints - 1)
        waypoint_times = np.arange(0, n_waypoints) * waypoint_time_step
        wpoints = np.zeros((n_waypoints, 3))
        #wpoints = np.zeros((n, 3))
        for i in range(3):
            wpoints[:, i] = np.interp(waypoint_times, rew_t, rewards[:, i])
            #wpoints[:, i] = rewards[:, i]
        waypoints.append(wpoints)
        n_stay = np.zeros(n_waypoints)

        # standardize data

        is_std = (inputs - obs_input_min[None, :]) / (obs_input_max[None, :] - obs_input_min[None, :] + eps)
        os_std = (outputs - obs_output_min[None, :]) / (obs_output_max[None, :] - obs_output_min[None, :] + eps)
        inps = np.array(inputs)  # -inf..inf
        inps = (inps - act_min[None, :]) / (act_max[None, :] - act_min[None, :] + 1e-10)  # 0..1
        inps = action_scale * (2 * inps - 1)  # -scale..scale

        # t start

        lb_time = (lookback - 1) * tstep
        idx0 = np.where(rew_t > lb_time)[0][1]

        # loop through trajectory points

        wp_first = wpoints[0, :]
        wp_last = wpoints[-1, :]

        for i in range(idx0, n - 1):

            dist_to_wps = np.linalg.norm(wpoints - rewards[i, :], axis=1)
            idx_sorted = np.argsort(dist_to_wps)
            wp_nearest_idx = idx_sorted[0]
            wp_nearest = wpoints[wp_nearest_idx, :]
            n_stay[idx_sorted[0]] += 1

            next_obs_dist_to_wps = np.linalg.norm(wpoints - rewards[i + 1, :], axis=1)
            next_obs_idx_sorted = np.argsort(next_obs_dist_to_wps)
            next_obs_wp_nearest_idx = next_obs_idx_sorted[0]
            next_obs_wp_nearest = wpoints[next_obs_wp_nearest_idx, :]

            # creating obs

            rp_with_lookback = np.zeros((lookback, rewards.shape[1]))
            is_std_with_lookback = np.zeros((lookback, inputs.shape[1]))
            os_std_with_lookback = np.zeros((lookback, outputs.shape[1]))
            for j in range(lookback):
                t = rew_t[i - 1] - j * tstep
                for k in range(rewards.shape[1]):
                    rp_with_lookback[j, k] = np.interp(t, rew_t, rewards[:, k])
                for k in range(inputs.shape[1]):
                    is_std_with_lookback[j, k] = np.interp(t, rew_t, is_std[:, k])
                for k in range(outputs.shape[1]):
                    os_std_with_lookback[j, k] = np.interp(t, rew_t, os_std[:, k])
            from_rp_to_nearest_wp_with_lookback = wp_nearest - rp_with_lookback
            from_rp_to_first_wp_with_lookback = wp_first - rp_with_lookback
            from_rp_to_last_wp_with_lookback = wp_last - rp_with_lookback
            traj = np.hstack([
                ((from_rp_to_first_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1),
                ((from_rp_to_nearest_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1),
                ((from_rp_to_last_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1)
            ])

            # reward

            reward_c1 = np.linalg.norm(wp_nearest - rewards[i, :])
            reward_c1 += reward_c1 / d_max
            reward_c2 = np.linalg.norm(wp_last - rewards[i, :])
            reward_c2 += reward_c2 / d_max
            reward = 1 - 0.5 * reward_c1 - 0.5 * reward_c2

            # creating next obs

            next_obs_rp_with_lookback = np.zeros((lookback, rewards.shape[1]))
            next_obs_is_std_with_lookback = np.zeros((lookback, inputs.shape[1]))
            next_obs_os_std_with_lookback = np.zeros((lookback, outputs.shape[1]))
            for j in range(lookback):
                t = rew_t[i] - j * tstep
                for k in range(rewards.shape[1]):
                    next_obs_rp_with_lookback[j, k] = np.interp(t, rew_t, rewards[:, k])
                for k in range(inputs.shape[1]):
                    next_obs_is_std_with_lookback[j, k] = np.interp(t, rew_t, is_std[:, k])
                for k in range(outputs.shape[1]):
                    next_obs_os_std_with_lookback[j, k] = np.interp(t, rew_t, os_std[:, k])
            next_obs_from_rp_to_nearest_wp_with_lookback = next_obs_wp_nearest - next_obs_rp_with_lookback
            next_obs_from_rp_to_first_wp_with_lookback = wp_first - next_obs_rp_with_lookback
            next_obs_from_rp_to_last_wp_with_lookback = wp_last - next_obs_rp_with_lookback
            next_obs = np.hstack([
                (next_obs_from_rp_to_first_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps),
                (next_obs_from_rp_to_nearest_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps),
                (next_obs_from_rp_to_last_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)
            ]).reshape(lookback, -1)

            # add signal values

            if use_inputs:
                traj = np.hstack([traj, is_std_with_lookback])
                next_obs = np.hstack([next_obs, next_obs_is_std_with_lookback])

            if use_outputs:
                traj = np.hstack([traj, os_std_with_lookback])
                next_obs = np.hstack([next_obs, next_obs_os_std_with_lookback])

            traj = traj.reshape(1, -1)

            # add actions

            traj = np.append(traj, inps[i, :])

            # add next obs

            traj = np.append(traj, next_obs)

            # add reward

            traj = np.append(traj, reward)

            # add to the list

            if len(trajectory_files) > 1:
                if ti < len(trajectory_files) - 1:
                    trajectories_tr.append(traj)
                else:
                    trajectories_val.append(traj)
            elif len(trajectory_files) == 1:
                trajectories_tr.append(traj)
                trajectories_val.append(traj)
            else:
                print('What??')
                raise NotImplemented

        n_stay_maxs.append(np.max(n_stay / n))

        last_dists.append(np.linalg.norm(wpoints[-1] - wpoints[-2]))

    n_stay_thr = np.maximum(np.mean(n_stay_maxs) + 3 * np.std(n_stay_maxs), np.max(n_stay_maxs))
    last_dist_thr = np.maximum(np.mean(last_dists) + 3 * np.std(last_dists), np.max(last_dists))

    return np.vstack(trajectories_tr), np.vstack(trajectories_val), waypoints, n_stay_thr, last_dist_thr