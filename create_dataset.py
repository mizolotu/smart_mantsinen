import os
import os.path as osp

from common.data_utils import prepare_trajectories, write_csv, write_json
from config import *

if __name__ == '__main__':

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir) if fpath.endswith('csv')]
    bc_train, bc_val, wps, traj_ids, traj_stages, wp_stages, wp_sizes = prepare_trajectories(
        signal_dir,
        trajectory_files,
        n_waypoints=nwaypoints,
        use_inputs=use_inputs,
        use_outputs=use_outputs,
        action_scale=action_scale,
        val_size=validation_size,
        wp_size=wp_size
    )
    print(bc_train.shape, bc_val.shape)

    # save dataset

    if not osp.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    write_csv(bc_train, dataset_dir, 'train.csv')
    write_csv(bc_val, dataset_dir, 'test.csv')
    write_json({
        'wp_sizes': wp_sizes,
        'wp_stages': wp_stages,
        'traj_stages': traj_stages,
        'traj_ids': traj_ids
    }, dataset_dir, 'metainfo.json')

    # save waypoints

    if not osp.isdir(waypoints_dir):
        os.mkdir(waypoints_dir)
    for i, wp in enumerate(wps):
        write_csv(wp, waypoints_dir, f'wps{i+1}.txt')