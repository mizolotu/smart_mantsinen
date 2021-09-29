import os
import os.path as osp

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from stable_baselines.ppo.ppod import PPOD as ppo
from stable_baselines.ppo.policies import MlpPolicy

from stable_baselines.common.vec_env.mevea_vec_env import MeveaVecEnv
from common.data_utils import prepare_trajectories
from config import *
from itertools import cycle

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

    # prepare training data

    trajectory_files = [osp.join(trajectory_dir, fpath) for fpath in os.listdir(trajectory_dir) if fpath.endswith('csv')]
    bc_train, bc_val, waypoints, n_stay_max, last_dist_max = prepare_trajectories(
        signal_dir,
        trajectory_files,
        use_inputs=use_inputs,
        use_outputs=use_outputs,
        action_scale=action_scale,
        lookback=lookback,
        tstep=tstep
    )
    n_stay_max *= nsteps

    # create environments

    n = len(waypoints)
    c = cycle(range(n))
    wp_inds = [next(c) for _ in range(nenvs)]
    env_fns = [
        make_env(
            MantsinenBasic,
            i,
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
            tstep,
            n_stay_max,
            last_dist_max,
            bonus
        ) for i in wp_inds
    ]
    env = MeveaVecEnv(env_fns)

    # create or load model

    chkp_dir = log_dir
    if not osp.isdir(chkp_dir):
        os.mkdir(chkp_dir)

    model = ppo(MlpPolicy, env, policy_kwargs=dict(net_arch = [256, 256, dict(vf=[64, 64]), dict(pi=[64, 64])]), batch_size=batch_size,
                n_steps=nsteps, model_path=chkp_dir, log_path=log_dir, tensorboard_log='tensorboard_log', verbose=1)

    if not model.loaded:
        model.pretrain(bc_train, data_val=bc_val, nepochs=npretrain)
        model.save(chkp_dir, 'first')

    # disable cuda

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # continue training

    model.learn(total_timesteps=ntrain)