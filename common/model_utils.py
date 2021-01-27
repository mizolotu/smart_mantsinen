import os, pathlib

from os import path as osp

def find_checkpoint_with_max_step(checkpoint_dir, prefix='rl_model_'):
    checkpoint_files = [item for item in os.listdir(checkpoint_dir) if osp.isfile(osp.join(checkpoint_dir, item)) and item.startswith(prefix) and item.endswith('.zip')]
    spl = [item.split(prefix)[1] for item in checkpoint_files]
    checkpoint_step_numbers = [int(item.split('_steps.zip')[0]) for item in spl]
    idx = sorted(range(len(checkpoint_step_numbers)), key=lambda k: checkpoint_step_numbers[k])
    return osp.join(checkpoint_dir, checkpoint_files[idx[-1]])

def find_checkpoint_with_latest_date(checkpoint_dir, prefix='rl_model_'):
    checkpoint_files = [item for item in os.listdir(checkpoint_dir) if osp.isfile(osp.join(checkpoint_dir, item)) and item.startswith(prefix) and item.endswith('.zip')]
    checkpoint_fpaths = [osp.join(checkpoint_dir, item) for item in checkpoint_files]
    checkpoint_dates = [pathlib.Path(item).stat().st_mtime for item in checkpoint_fpaths]
    idx = sorted(range(len(checkpoint_dates)), key=lambda k: checkpoint_dates[k])
    return checkpoint_fpaths[idx[-1]]

def clean_checkpoint_directory(checkpoint_dir, exclude=[], prefix='rl_model_'):
    checkpoint_files = [item for item in os.listdir(checkpoint_dir) if osp.isfile(osp.join(checkpoint_dir, item)) and item.startswith(prefix) and item.endswith('.zip')]
    checkpoint_fpaths = [osp.join(checkpoint_dir, item) for item in checkpoint_files]
    for fpath in checkpoint_fpaths:
        if fpath not in exclude:
            print('Deleting {0}'.format(fpath))
            os.remove(fpath)


