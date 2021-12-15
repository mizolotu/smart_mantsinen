import os
import os.path as osp

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from stable_baselines.ppo.ppod import PPOD as ppo
from stable_baselines.ppo.policies import MlpPolicy

from stable_baselines.common.vec_env.mevea_vec_env import MeveaVecEnv
from common.data_utils import read_csv, load_waypoints_and_meta, load_signals
from config import *

def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    # configure logger

    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    log_dir = osp.join(os.path.abspath(model_output), 'ppo')

    # check that server is running

    while not is_server_running(server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # load waypoints and meta

    tr_waypoints, te_waypoints, tr_traj_sizes, te_traj_sizes = load_waypoints_and_meta(waypoints_dir, dataset_dir)

    # default action

    _, xmin, xmax = load_signals(signal_dir, 'input')
    default_action = [(x - y) / (z - y) for x, y, z in zip(default_actions[1], xmin, xmax)]

    # create environments

    env_fns = [
        make_env(
            MantsinenBasic,
            i,
            model_path,
            model_dir,
            signal_dir,
            server,
            tr_waypoints,
            nsteps,
            lookback,
            use_inputs,
            use_outputs,
            action_scale,
            wp_size,
            tstep,
            bonus
        ) for i in range(nenvs)
    ] + [
        make_env(
            MantsinenBasic,
            nenvs,
            model_path,
            model_dir,
            signal_dir,
            server,
            te_waypoints,
            nsteps,
            lookback,
            use_inputs,
            use_outputs,
            action_scale,
            wp_size,
            tstep,
            bonus
        )
    ]
    env = MeveaVecEnv(env_fns)

    # create or load model

    chkp_dir = log_dir
    if not osp.isdir(chkp_dir):
        os.mkdir(chkp_dir)

    model = ppo(MlpPolicy, env, n_env_train=nenvs, npoints=npoints, lookback=lookback, policy_kwargs=dict(net_arch = ppo_net_arch), batch_size=batch_size,
                n_steps=nsteps, model_path=chkp_dir, log_path=log_dir, tensorboard_log='tensorboard_log', verbose=1)

    if not model.loaded:
        bc_train = read_csv(dataset_dir, 'train.csv')
        bc_val = read_csv(dataset_dir, 'test.csv')
        model.pretrain(bc_train, bc_val, tr_traj_sizes, te_traj_sizes, tstep, default_action=default_action, nepochs=npretrain)
        model.save(chkp_dir, 'first')

    # disable cuda

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # continue training

    model.learn(total_timesteps=ntrain)