import os, pandas
import os.path as osp
import numpy as np

from common.data_utils import load_signals, load_trajectory, update_signals

if __name__ == '__main__':

    # parameters

    trajectory_dir = 'data/trajectory_examples'
    signal_dir = 'data/signals'

    # load signals

    signals = {}
    mins = {}
    maxs = {}
    for key in ['input', 'output', 'reward']:
        values = load_signals(signal_dir, key)
        signals[key] = values[0]
        if len(values) == 3:
            mins[key] = values[1]
            maxs[key] = values[2]
        else:
            mins[key] = []
            maxs[key] = []

    # find min and max values for each element

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir)]
    assert len(trajectory_files) > 0
    names = pandas.read_csv(trajectory_files[0], header=None, nrows=1).values[0].tolist()
    values = []
    for trajectory_file in trajectory_files:
        t, inputs, outputs, rewards = load_trajectory(trajectory_file, signals)
        values.append(np.hstack([rewards, inputs, outputs]))
    values = np.vstack(values)

    xmins = np.min(values, axis=0)
    xmaxs = np.max(values, axis=0)

    # read signals

    for key in signals.keys():
        values, value_mins, value_maxs = signals[key], mins[key], maxs[key]
        if len(values) == len(value_mins) and len(values) == len(value_maxs):
            pass
        elif len(values) > 0 and len(value_maxs) == 0 and len(value_mins) == 0:
            value_mins = np.zeros_like(values, dtype=float)
            value_maxs = np.zeros_like(values, dtype=float)
            for i in range(len(values)):
                idx = names.index(values[i])- 1
                value_mins[i] = xmins[idx]
                value_maxs[i] = xmaxs[idx]
            update_signals(signal_dir, key, values, value_mins, value_maxs)
        else:
            raise NotImplemented
