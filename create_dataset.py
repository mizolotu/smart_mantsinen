import os
import os.path as osp

from common.data_utils import prepare_trajectories, write_csv, write_json
from config import *

if __name__ == '__main__':

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir) if fpath.endswith('csv')]
    bc_train, bc_val, wps, wp_ids, wp_stages, traj_sizes, n_stay_max = prepare_trajectories(
        signal_dir,
        trajectory_files,
        n_waypoints=nwaypoints,
        obs_wp_freq=obs_wp_freq,
        use_inputs=use_inputs,
        use_outputs=use_outputs,
        action_scale=action_scale,
        val_size=validation_size,
        wp_size=wp_size
    )
    n_stay_max *= nsteps
    print(bc_train.shape, bc_val.shape)

    # save dataset

    if not osp.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    write_csv(bc_train, dataset_dir, 'train.csv')
    write_csv(bc_val, dataset_dir, 'test.csv')
    write_json({
        'n_stay_max': n_stay_max,
        'wp_ids': wp_ids,
        'wp_stages': wp_stages,
        'traj_sizes': traj_sizes
    }, dataset_dir, 'metainfo.json')

    # save waypoints

    if not osp.isdir(waypoints_dir):
        os.mkdir(waypoints_dir)
    for i, wp in enumerate(wps):
        write_csv(wp, waypoints_dir, f'wps{i+1}.txt')