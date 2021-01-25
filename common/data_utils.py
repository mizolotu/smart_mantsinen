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

def save_trajectory(signal_caps, timestamps, rewards, states, stay_t_start, output):
    timestamps = np.array(timestamps)
    rewards = np.array(rewards)
    states = np.array(states)
    idx = np.where(timestamps < stay_t_start)[0]
    header = np.hstack(['Timestamp', signal_caps['reward'], signal_caps['input'], signal_caps['output']])
    trajectory = np.hstack([timestamps[idx, None] - timestamps[idx[0]], rewards[idx, :], states[idx, :-3]])
    arr = np.vstack([header, trajectory])
    pandas.DataFrame(arr).to_csv(output, index=False, header=False)

def load_trajectory(fpath):
    p = pandas.read_csv(fpath, header=0)
    v = p.values
    t = v[:, 0]
    xyz = v[:, 1:]
    return t, xyz
