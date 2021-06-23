import argparse, os, sys
import os.path as osp
import numpy as np

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from baselines.sac.sac import SAC as sac
from baselines.sac.policies import MlpPolicy
from common.mevea_vec_env import MeveaVecEnv
from common.model_utils import find_checkpoint_with_latest_date
from common.data_utils import prepare_trajectories
from common.callbacks import CheckpointCallback
from common import logger
from config import *

def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Checkpoint file.')  # e.g. "models/mevea/mantsinen/ppo/model_checkpoints/rl_model_5001216_steps.zip"
    parser.add_argument('-s', '--save', type=bool, help='Save new training steps?', default=True)
    args = parser.parse_args()

    # configure logger

    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    log_dir = osp.join(os.path.abspath(model_output), 'sac')
    logger.configure(log_dir, format_strs)

    # check that server is running

    while not is_server_running(server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir) if fpath.endswith('csv')]
    bc_train, bc_val, waypoints = prepare_trajectories(
        signal_dir,
        trajectory_files,
        use_inputs=use_inputs,
        use_outputs=use_outputs,
        action_scale=action_scale,
        lookback=lookback,
        tstep=tstep
    )
    print(bc_train.shape, bc_val.shape)

    # create environments

    env_fns = [make_env(
        MantsinenBasic,
        model_path,
        model_dir,
        signal_dir,
        server,
        waypoints,
        nsteps,
        lookback,
        use_inputs,
        use_outputs,
        action_scale,
        tstep
    )]
    env = MeveaVecEnv(env_fns)

    try:

        # load model

        if args.checkpoint is None:
            checkpoint_file = find_checkpoint_with_latest_date('{0}/model_checkpoints/'.format(log_dir))
        else:
            checkpoint_file = args.checkpoint
        model = sac.load(checkpoint_file)
        model.set_env(env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        # create and pretrain model

        model = sac(MlpPolicy, env, n_steps=nsteps, verbose=1)
        model.full_pretrain(np.vstack([bc_train, bc_val]), batch_size=nsteps, n_epochs=100)
        if not osp.isdir(f'{log_dir}/model_checkpoints/'):
            os.mkdir(f'{log_dir}/model_checkpoints/')
        model.save(f'{log_dir}/model_checkpoints/rl_model_0_steps.zip')

    callbacks = []
    if args.save:
        callbacks.append(CheckpointCallback(save_freq=nsteps, save_path='{0}/model_checkpoints/'.format(log_dir)))
    model.learn(total_timesteps=ntrain, callback=callbacks)