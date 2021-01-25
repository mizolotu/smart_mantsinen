import argparse, os
import os.path as osp
import numpy as np

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running, is_backend_registered, post_signals, get_state
from time import sleep
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.policies import MlpPolicy
from common.mevea_vec_env import MeveaVecEnv
from common.runners import MeveaRunner

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
    parser.add_argument('-m', '--model', help='File path to the model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-t', '--trajectories', help='Trajectory files.', default='trajectory1.csv,trajectory2.csv,trajectory3.csv,trajectory4.csv')
    parser.add_argument('-f', '--frequency', help='Action frequency.', default=1, type=float)
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

    model = ppo(MlpPolicy, env, MeveaRunner, verbose=1)
    model.learn(total_timesteps=nsteps)