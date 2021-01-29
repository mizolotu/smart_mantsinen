import argparse, pandas
import os.path as osp
import numpy as np

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.policies import MlpPolicy, CnnPolicy
from common.mevea_vec_env import MeveaVecEnv
from common.runners import MeveaRunner
from common.data_utils import prepare_trajectories, load_signals

def make_env(env_class, mevea_model, signal_dir, trajectory_data, server_url, frequency, use_signals):
    fn = lambda: env_class(mevea_model, signal_dir, trajectory_data, server_url, frequency, use_signals)
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
    parser.add_argument('-f', '--frequency', help='Action frequency.', default=0.1, type=float)
    parser.add_argument('-o', '--output', help='Result output.', default='models/mevea/mantsinen/ppo')
    args = parser.parse_args()

    # check that server is running

    while not is_server_running(args.server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in args.trajectories.split(',')]
    trajectory_data = prepare_trajectories(signal_dir, trajectory_files, args.frequency, use_signals=use_signals)

    # create environments

    env_fns = [make_env(MantsinenBasic, args.model, signal_dir, data, args.server, args.frequency, use_signals) for data in trajectory_data]
    env = MeveaVecEnv(env_fns)

    # create model

    model = ppo(MlpPolicy, env, runner=MeveaRunner, verbose=1)

    # pretrain model

    model.pretrain(trajectory_data, n_epochs=1000)

    # check actions after pretraining for validation purposes

    input_signals, mins, maxs = load_signals(signal_dir, 'input')
    mins = np.array(mins)
    maxs = np.array(maxs)
    header = ['Timestamp', *input_signals]
    actions = [header]
    traj = trajectory_data[0]
    actions = [header]
    traj = trajectory_data[0]
    for i in range(len(traj)):
        t = traj[i, -1]
        obs = traj[i, :model.env.observation_space.shape[0]]
        a = model.predict(obs)[0]
        a = (a + 1) / 2
        a = a * (maxs - mins) + mins
        actions.append(np.hstack([t, a]))

    #  save actions

    output = 'tmp/actions.csv'
    pandas.DataFrame(np.vstack(actions)).to_csv(output, index=False, header=False)

