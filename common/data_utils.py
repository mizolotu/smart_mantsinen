import pandas, json, os
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

def prepare_trajectories(signal_dir, trajectory_files, n_waypoints, obs_wp_freq, use_inputs=True, use_outputs=True, action_scale=1, val_size=0.4, seed=0, wp_size=1):

    # set seed

    np.random.seed(seed)

    # load signals and their limits

    signals, mins, maxs = {}, {}, {}
    for key in ['input', 'output', 'reward']:
        values, xmin, xmax = load_signals(signal_dir, key)
        signals[key] = values
        mins[key] = xmin
        maxs[key] = xmax

    # set standardization vectors

    rew_min, rew_max = np.array(mins['reward']), np.array(maxs['reward'])
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
    waypoint_stages = []
    waypoint_traj_ids = []
    traj_sizes = []
    n_stay_maxs = []
    last_dists = []
    rews = []
    val_size = len(trajectory_files) * val_size
    val_size = 1 if val_size < 1 else np.floor(val_size)

    trajectory_file_ids = np.arange(len(trajectory_files))
    np.random.shuffle(trajectory_file_ids)
    for ei, ti in enumerate(trajectory_file_ids):
        fpath = trajectory_files[ti]

        # load trajectories

        rew_t, inputs, outputs, rewards = load_trajectory(fpath, signals)
        rews.append(rewards)

        # calculate way points by interpolating over time

        n = len(rew_t)
        duration = rew_t[-1] - rew_t[0]
        waypoint_time_step = duration / (n_waypoints - 1)
        waypoint_times = np.arange(0, n_waypoints) * waypoint_time_step
        wpoints = np.zeros((n_waypoints, 3))
        for i in range(3):
            wpoints[:, i] = np.interp(waypoint_times, rew_t, rewards[:, i])
        waypoints.append(wpoints)
        n_stay = np.zeros(n_waypoints)
        waypoints_completed = np.zeros(n_waypoints)

        # standardize data

        is_std = (inputs - obs_input_min[None, :]) / (obs_input_max[None, :] - obs_input_min[None, :] + eps)
        os_std = (outputs - obs_output_min[None, :]) / (obs_output_max[None, :] - obs_output_min[None, :] + eps)
        inps = np.array(inputs)  # -inf..inf
        inps = (inps - act_min[None, :]) / (act_max[None, :] - act_min[None, :] + 1e-10)  # 0..1
        inps_scaled = action_scale * (2 * inps - 1)  # -scale..scale

        # loop through trajectory points

        wp_first = wpoints[0, :]
        wp_last = wpoints[-1, :]

        traj_full = []
        tmin = rew_t[0]
        for i in range(n - 1):

            wps_not_completed_idx = np.where(waypoints_completed == 0)[0]
            if len(wps_not_completed_idx) > 0:
                dist_to_wps = np.linalg.norm(wpoints - rewards[i, :], axis=1)
                idx_min_all = np.argmin(dist_to_wps)
                idx_min_not_completed = np.argmin(dist_to_wps[wps_not_completed_idx])
                wp_nearest = wpoints[idx_min_all, :]
                wp_nearest_not_completed = wpoints[wps_not_completed_idx[idx_min_not_completed], :]
                n_stay[idx_min_all] += 1
            else:
                wp_nearest_not_completed = wpoints[-1, :]

            # update completed wps

            if np.linalg.norm(wp_nearest - rewards[i, :]) <= wp_size:
                wps_not_completed_idx[idx_min_all] = 1

            # creating obs

            rp_with_lookback = np.zeros((1, rewards.shape[1]))
            is_std_with_lookback = np.zeros((1, inputs.shape[1]))
            os_std_with_lookback = np.zeros((1, outputs.shape[1]))
            t = rew_t[i]
            for k in range(rewards.shape[1]):
                rp_with_lookback[0, k] = np.interp(t, rew_t, rewards[:, k])
            for k in range(inputs.shape[1]):
                is_std_with_lookback[0, k] = np.interp(t, rew_t, is_std[:, k])
            for k in range(outputs.shape[1]):
                os_std_with_lookback[0, k] = np.interp(t, rew_t, os_std[:, k])

            from_rp_to_nearest_wp_with_lookback = wp_nearest_not_completed - rp_with_lookback
            from_rp_to_nearest_wp_with_lookback_norm = np.linalg.norm(from_rp_to_nearest_wp_with_lookback, 2, 1)
            from_rp_to_nearest_wp_with_lookback /= (from_rp_to_nearest_wp_with_lookback_norm[:, None] + 1e-10)
            from_rp_to_nearest_wp_with_lookback_norm_std = from_rp_to_nearest_wp_with_lookback_norm / d_max

            #from_rp_to_first_wp_with_lookback = wp_first - rp_with_lookback

            from_rp_to_last_wp_with_lookback = wp_last - rp_with_lookback
            from_rp_to_last_wp_with_lookback_norm = np.linalg.norm(from_rp_to_last_wp_with_lookback, 2, 1)
            from_rp_to_last_wp_with_lookback /= (from_rp_to_last_wp_with_lookback_norm[:, None] + 1e-10)
            from_rp_to_last_wp_with_lookback_norm_std = from_rp_to_last_wp_with_lookback_norm / d_max

            from_rp_to_first_wp_with_lookback = wp_first - rp_with_lookback
            from_rp_to_first_wp_with_lookback_norm = np.linalg.norm(from_rp_to_first_wp_with_lookback, 2, 1)
            from_rp_to_first_wp_with_lookback /= (from_rp_to_first_wp_with_lookback_norm[:, None] + 1e-10)
            from_rp_to_first_wp_with_lookback_norm_std = from_rp_to_first_wp_with_lookback_norm / d_max

            obs_wp_freq = np.clip(obs_wp_freq, 1, n_waypoints)


            from_rp_to_wps_with_lookback = [
                from_rp_to_first_wp_with_lookback,
                from_rp_to_first_wp_with_lookback_norm_std.reshape(1, 1),
                from_rp_to_nearest_wp_with_lookback,
                from_rp_to_nearest_wp_with_lookback_norm_std.reshape(1, 1),
                from_rp_to_last_wp_with_lookback,
                from_rp_to_last_wp_with_lookback_norm_std.reshape(1, 1)
            ]
            # + [
                #wpoints[i, :] - rp_with_lookback for i in range(n_waypoints - 1, -1, -obs_wp_freq)
            #]

            #traj = np.hstack([
            #    ((from_rp_to_first_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1),
            #    ((from_rp_to_nearest_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1),
            #    ((from_rp_to_last_wp_with_lookback.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1)
            #])

            traj = np.hstack(
                #[((wp.reshape(1, -1)[0] - v_min) / (v_max - v_min + eps)).reshape(lookback, -1) for wp in from_rp_to_wps_with_lookback]
                from_rp_to_wps_with_lookback
            )
            #print(traj.shape)

            # add signal values

            if use_inputs:
                traj = np.hstack([traj, is_std_with_lookback])

            if use_outputs:
                traj = np.hstack([traj, os_std_with_lookback])

            traj = traj.reshape(1, -1)

            # add actions

            traj = np.append(traj, inps_scaled[i, :])

            # add timestamp

            traj = np.append(traj, [t - tmin])

            # add to full traj

            traj_full.append(traj)

        traj_full = np.vstack(traj_full)

        # add to the lists

        waypoint_traj_ids.append(int(ti))
        traj_sizes.append(len(traj_full))
        if len(trajectory_files) > 1:
            if ei < len(trajectory_files) - val_size:
                trajectories_tr.append(traj_full)
                waypoint_stages.append('train')
            else:
                trajectories_val.append(traj_full)
                waypoint_stages.append('test')
        else:
            raise NotImplemented

        n_stay_maxs.append(np.max(n_stay / n))

        last_dists.append(np.linalg.norm(wpoints[-1] - wpoints[-2]))

    n_stay_thr = np.maximum(np.mean(n_stay_maxs) + 3 * np.std(n_stay_maxs), np.max(n_stay_maxs))

    return np.vstack(trajectories_tr), np.vstack(trajectories_val), waypoints, waypoint_traj_ids, waypoint_stages, traj_sizes, n_stay_thr

def load_waypoints_and_meta(waypoints_dir, dataset_dir):
    wp_files = [osp.join(waypoints_dir, fpath) for fpath in os.listdir(waypoints_dir) if fpath.endswith('txt')]
    waypoints = []
    for i, wp in enumerate(wp_files):
        waypoints.append(read_csv(waypoints_dir, f'wps{i+1}.txt'))
    meta = read_json(dataset_dir, 'metainfo.json')
    tr_waypoints = [wp for wp, wp_stage in zip(waypoints, meta['wp_stages']) if wp_stage == 'train']
    tr_traj_sizes = [s for s, wp_stage in zip(meta['traj_sizes'], meta['wp_stages']) if wp_stage == 'train']
    te_waypoints = [wp for wp, wp_stage in zip(waypoints, meta['wp_stages']) if wp_stage == 'test']
    te_traj_sizes = [s for s, wp_stage in zip(meta['traj_sizes'], meta['wp_stages']) if wp_stage == 'test']
    return tr_waypoints, te_waypoints, tr_traj_sizes, te_traj_sizes

def get_test_waypoints(fname):
    v = pandas.read_csv(fname, header=None).values
    assert v.shape[0] > 1, 'Not enough points, should be at least two!'
    return v

def write_csv(x, fdir, fname):
    pandas.DataFrame(x).to_csv(osp.join(fdir, fname), index=False, header=False)

def read_csv(fdir, fname):
    return pandas.read_csv(osp.join(fdir, fname), header=None).values

def write_json(data, fdir, fname):
    with open(osp.join(fdir, fname), 'w') as f:
        json.dump(data, f)

def read_json(fdir, fname):
    with open(osp.join(fdir, fname), 'r') as f:
        data = json.load(f)
    return data
