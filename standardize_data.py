import os, pandas
import os.path as osp
import numpy as np

from common.data_utils import load_signals, load_trajectory, update_signals

from time import sleep

if __name__ == '__main__':

    # parameters

    trajectory_dir = 'data/trajectory_examples'
    signal_csv = {
        'input': 'data/signals/input.csv',
        'output': 'data/signals/output.csv',
        'reward': 'data/signals/reward.csv'
    }

    # find min and max values for each element

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir)]
    assert len(trajectory_files) > 0
    names = pandas.read_csv(trajectory_files[0], header=None, nrows=1).values[0].tolist()
    values = []
    for trajectory_file in trajectory_files:
        t, xyz = load_trajectory(trajectory_file)
        values.append(xyz)
    values = np.vstack(values)

    xmins = np.min(values, axis=0)
    xmaxs = np.max(values, axis=0)

    # read signals

    for key in signal_csv.keys():
        values, value_mins, value_maxs = load_signals(signal_csv[key])
        if len(values) == len(value_mins) and len(values) == len(value_maxs):
            for i in range(len(values)):
                idx = names.index(values[i]) - 1
                value_mins[i] = np.minimum(xmins[idx], value_mins[i])
                value_maxs[i] = np.maximum(xmaxs[idx], value_maxs[i])
            update_signals(signal_csv[key], values, np.array(value_mins), np.array(value_maxs))
        elif len(values) > 0 and len(value_maxs) == 0 and len(value_mins) == 0:
            value_mins = np.zeros_like(values, dtype=float)
            value_maxs = np.zeros_like(values, dtype=float)
            for i in range(len(values)):
                idx = names.index(values[i])- 1
                value_mins[i] = xmins[idx]
                value_maxs[i] = xmaxs[idx]
            update_signals(signal_csv[key], values, value_mins, value_maxs)
        else:
            raise NotImplemented
