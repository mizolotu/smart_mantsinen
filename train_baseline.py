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

def make_env(env_class, mevea_model, signal_dir, trajectory_data, server_url, delay, use_signals):
    fn = lambda: env_class(mevea_model, signal_dir, trajectory_data, server_url, delay, use_signals)
    return fn

if __name__ == '__main__':

    # parameters

    trajectory_dir = 'data/trajectory_examples'
    signal_dir = 'data/signals'

    nsteps = 100000000
    sleep_interval = 3
    use_signals = False

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the Mevea model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-t', '--trajectories', help='Trajectory files.', default='trajectory1.csv,trajectory2.csv,trajectory3.csv,trajectory4.csv')
    parser.add_argument('-d', '--delay', help='Action delay.', default=0.1, type=float)
    parser.add_argument('-o', '--output', help='Result output.', default='models/mevea/mantsinen/ppo')
    args = parser.parse_args()

    # check that server is running

    while not is_server_running(args.server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in args.trajectories.split(',')]
    trajectory_data = prepare_trajectories(signal_dir, trajectory_files, args.delay, use_signals=use_signals)

    # create environments

    env_fns = [make_env(MantsinenBasic, args.model, signal_dir, data, args.server, args.delay, use_signals) for data in trajectory_data]
    env = MeveaVecEnv(env_fns)

    try:

        # load model

        checkpoint_file = find_checkpoint_with_max_step('{0}/model_checkpoints/'.format(args.output))
        model = ppo.load(checkpoint_file)
        model.set_env(env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))

    except Exception as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

        # create and pretrain model

        model = ppo(MlpPolicy, env, runner=MeveaRunner, verbose=1)
        model.pretrain(trajectory_data, n_epochs=10000)
        model.save('{0}/model_checkpoints/rl_model_0_steps.zip'.format(args.output))

    finally:

        # continue training

        checkpoint_callback = CheckpointCallback(save_freq=2048, save_path='{0}/model_checkpoints/'.format(args.output))
        model.learn(total_timesteps=nsteps, callback=[checkpoint_callback])