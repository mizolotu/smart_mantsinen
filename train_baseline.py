import argparse, os, sys
import os.path as osp
import numpy as np

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.policies import MlpPolicy
from common.mevea_vec_env import MeveaVecEnv
from common.mevea_runner import MeveaRunner
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
    parser.add_argument('-g', '--goon', type=bool, help='Continue training?', default=True)
    args = parser.parse_args()

    # configure logger

    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(model_output), format_strs)

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

    # create environments

    n = len(waypoints)
    wp_inds = np.random.choice(n, nenvs)
    env_fns = [
        make_env(
            MantsinenBasic,
            model_path,
            model_dir,
            signal_dir,
            server,
            waypoints[i],
            nsteps,
            lookback,
            use_inputs,
            use_outputs,
            action_scale,
            tstep
        ) for i in wp_inds
    ]
    env = MeveaVecEnv(env_fns)

    try:

        # load model

        if args.checkpoint is None:
            checkpoint_file = find_checkpoint_with_latest_date('{0}/model_checkpoints/'.format(model_output))
        else:
            checkpoint_file = args.checkpoint
        model = ppo.load(checkpoint_file)
        model.set_env(env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        # create and pretrain model

        model = ppo(MlpPolicy, env, runner=MeveaRunner, n_steps=nsteps, verbose=1)
        model.pretrain(bc_train, bc_val, n_epochs=npretrain, log_freq=npretrain)
        model.save('{0}/model_checkpoints/rl_model_0_steps.zip'.format(model_output))

    finally:

        # continue training

        callbacks = []
        if args.goon:
            callbacks.append(CheckpointCallback(save_freq=nsteps, save_path='{0}/model_checkpoints/'.format(model_output)))
        model.learn(total_timesteps=ntrain, callback=callbacks)