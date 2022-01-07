import os
import os.path as osp

from common.data_utils import prepare_trajectories, write_csv, write_json
from config import *

if __name__ == '__main__':

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir) if fpath.endswith('csv')]
    bc_train, bc_val, bc_inf, wps, traj_ids, traj_stages, traj_sizes = prepare_trajectories(
        signal_dir,
        trajectory_files,
        n_waypoints=nwaypoints,
        action_scale=action_scale,
        val_size=validation_size,
        inf_size=inference_size,
        wp_size=wp_size,
        use_inputs=use_inputs
    )
    print(bc_train.shape, bc_val.shape, bc_inf.shape)

    # save dataset

    if not osp.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    write_csv(bc_train, dataset_dir, 'tr.csv')
    write_csv(bc_val, dataset_dir, 'val.csv')
    write_csv(bc_inf, dataset_dir, 'inf.csv')
    write_json({
        'traj_sizes': traj_sizes,
        'traj_stages': traj_stages,
        'traj_ids': traj_ids
    }, dataset_dir, 'metainfo.json')

    # save waypoints

    if not osp.isdir(waypoints_dir):
        os.mkdir(waypoints_dir)
    for i, wp in enumerate(wps):
        write_csv(wp, waypoints_dir, f'wps{i+1}.txt')