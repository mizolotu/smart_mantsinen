import argparse, os, sys
import os.path as osp

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.policies import MlpPolicy
from common.mevea_vec_env import MeveaVecEnv
from common.mevea_runner import MeveaRunner
from common.model_utils import find_checkpoint_with_max_step
from common.data_utils import prepare_trajectories
from common.callbacks import CheckpointCallback

def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    # parameters

    trajectory_dir = 'data/trajectory_examples'
    signal_dir = 'data/signals'

    nsteps = 100000000
    sleep_interval = 3
    use_signals = True
    action_scale = 100
    npoints = 128
    lookback = 4
    wp_dist = 1

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the Mevea model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-t', '--trajectories', help='Trajectory files.', default='trajectory1.csv,trajectory2.csv,trajectory3.csv,trajectory4.csv')
    parser.add_argument('-o', '--output', help='Result output.', default='models/mevea/mantsinen/ppo')
    parser.add_argument('-c', '--checkpoint', help='Checkpoint file.', default='models/mevea/mantsinen/ppo/model_checkpoints/rl_model_0_steps.zip')
    parser.add_argument('-g', '--goon', type=bool, help='Continue training?', default=True)
    args = parser.parse_args()

    # check that server is running

    while not is_server_running(args.server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in args.trajectories.split(',')]
    bc_data, waypoints, bonus = prepare_trajectories(signal_dir, trajectory_files, use_signals=use_signals, action_scale=action_scale, lookback=lookback, wp_dist=wp_dist)

    # create environments

    env_fns = [make_env(MantsinenBasic, args.model, signal_dir, args.server, data, lookback, use_signals, action_scale, wp_dist, bonus) for data in waypoints]
    env = MeveaVecEnv(env_fns)

    try:

        # load model

        if args.checkpoint == '':
            checkpoint_file = find_checkpoint_with_max_step('{0}/model_checkpoints/'.format(args.output))
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

        model = ppo(MlpPolicy, env, runner=MeveaRunner, nminibatches=len(waypoints), verbose=1)
        model.pretrain(bc_data, n_epochs=10000, log_freq=10000)
        model.save('{0}/model_checkpoints/rl_model_0_steps.zip'.format(args.output))

    finally:

        # continue training
        callbacks = []
        if args.goon:
            callbacks.append(CheckpointCallback(save_freq=2048, save_path='{0}/model_checkpoints/'.format(args.output)))

        model.learn(total_timesteps=nsteps, callback=callbacks)