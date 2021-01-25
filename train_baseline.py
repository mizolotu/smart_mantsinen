import argparse, os, sys
import os.path as osp
import numpy as np

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running, is_backend_registered, post_signals, get_state
from time import sleep
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.policies import MlpPolicy
from common.mevea_vec_env import MeveaVecEnv
from common.runners import MeveaRunner
from common.model_utils import find_checkpoint_with_max_step
from common.callbacks import CheckpointCallback

def make_env(env_class, mevea_model, signal_csv, identical_signals, trajectory_csv, server_url, frequency):
    fn = lambda: env_class(mevea_model, signal_csv, identical_signals, trajectory_csv, server_url, frequency)
    return fn

if __name__ == '__main__':

    # parameters

    trajectory_dir = 'data/trajectory_examples'
    signal_csv = {
        'input': 'data/signals/input.csv',
        'output': 'data/signals/output.csv',
        'reward': 'data/signals/reward.csv'
    }
    identical_signals = {
        'AI_UnderCar_Bus_SimSta1_u16Boom1Cur': ['AI_UnderCar_Bus_SimSta1_u16Boom2Cur'],
        'AI_UnderCar_Bus_SimSta3_u16Stick1Cur': ['AI_UnderCar_Bus_SimSta3_u16Stick1Cur_Y15', 'AI_UnderCar_Bus_SimSta3_u16Stick2Cur']
    }
    nsteps = 100000000
    sleep_interval = 3

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the Mevea model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-t', '--trajectories', help='Trajectory files.', default='trajectory1.csv,trajectory2.csv,trajectory3.csv,trajectory4.csv')
    parser.add_argument('-f', '--frequency', help='Action frequency.', default=0, type=float)
    parser.add_argument('-o', '--output', help='Result output.', default='models/mevea/mantsinen/ppo')
    args = parser.parse_args()

    # check that server is running

    while not is_server_running(args.server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # create environments

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in args.trajectories.split(',')]
    env_fns = [make_env(MantsinenBasic, args.model, signal_csv, identical_signals, fpath, args.server, args.frequency) for fpath in trajectory_files]
    env = MeveaVecEnv(env_fns)

    # create model

    try:
        checkpoint_file = find_checkpoint_with_max_step('{0}/checkpoints/'.format(args.output))
        model = ppo.load(checkpoint_file)
        model.set_env(env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        model = ppo(MlpPolicy, env, runner=MeveaRunner, verbose=1)
    finally:
        checkpoint_callback = CheckpointCallback(save_freq=2048, save_path='{0}/model_checkpoints/'.format(args.output))
        model.learn(total_timesteps=nsteps, callback=[checkpoint_callback])

    model = ppo(MlpPolicy, env, MeveaRunner, verbose=1)
    model.learn(total_timesteps=nsteps)