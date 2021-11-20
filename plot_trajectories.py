import numpy as np
import os.path as osp
import os

from config import trajectory_dir, signal_dir, trajectory_plot_dir, nwaypoints
from common.data_utils import load_signals, load_trajectory
from common.plot_utils import plot_multiple_lines

if __name__ == '__main__':

    plot_labels = ['x', 'y', 'z']

    # load signals

    signals = {}
    for key in ['input', 'output', 'reward']:
        values, _, _ = load_signals(signal_dir, key)
        signals[key] = values

    # trajectory files

    trajectory_files = [fname for fname in os.listdir(trajectory_dir) if fname.endswith('.csv')]

    for trajectory_file in trajectory_files:

        rew_t, inputs, outputs, rewards = load_trajectory(osp.join(trajectory_dir, trajectory_file), signals)

        n = len(rew_t)
        duration = rew_t[-1] - rew_t[0]
        waypoint_time_step = duration / (nwaypoints - 1)
        waypoint_times = np.arange(0, nwaypoints) * waypoint_time_step
        wpoints = np.zeros((nwaypoints, 3))
        for i in range(3):
            wpoints[:, i] = np.interp(waypoint_times, rew_t, rewards[:, i])

        # plot

        for i in range(rewards.shape[1]):
            plot_multiple_lines([rew_t, waypoint_times], [rewards[:, i], wpoints[:, i]], ['b', 'ro'], 'Time', 'Value', osp.join(trajectory_plot_dir, f"{trajectory_file.split('csv')[0]}_{plot_labels[i]}.pdf"))