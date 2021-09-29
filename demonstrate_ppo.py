import argparse
import os.path as osp
import numpy as np

from env_frontend import MantsinenBasic
from common.server_utils import is_server_running
from time import sleep
from stable_baselines.ppo.ppod import PPOD as ppo
from stable_baselines.ppo.policies import PPOPolicy as policy
from stable_baselines.common.vec_env.mevea_vec_env import MeveaVecEnv
from common.data_utils import get_test_waypoints
from config import *

def make_env(env_class, *args):
    fn = lambda: env_class(*args)
    return fn

if __name__ == '__main__':

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--waypoints', help='Text file with waypoints.', default='example_waypoints.txt')
    parser.add_argument('-m', '--model', help='Model directory.', default='models/mevea/mantsinen/ppo')
    parser.add_argument('-c', '--checkpoint', help='Checkpoint', default='best', choices=['first', 'last', 'best'])
    parser.add_argument('-v', '--video', help='Record video?', type=bool)
    args = parser.parse_args()

    chkpt_dir = args.model

    # check that server is running

    while not is_server_running(server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # extract waypoints

    waypoints = get_test_waypoints(args.waypoints)
    last_dist_max = np.linalg.norm(waypoints[-1] - waypoints[-2])
    n_stay_max = np.inf

    # create environment

    env_fns = [make_env(
        MantsinenBasic,
        0,
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
        tstep,
        n_stay_max,
        last_dist_max,
        bonus
    )]
    env = MeveaVecEnv(env_fns)

    # load model and run it in demo mode

    try:
        model = ppo(policy, env, policy_kwargs=dict(net_arch=[256, 256, dict(vf=[64, 64]), dict(pi=[64, 64])]), batch_size=batch_size, n_steps=nsteps,
                    model_path=chkpt_dir, chkpt_name=args.checkpoint, tensorboard_log='tensorboard_log', verbose=1)
        if args.video:
            cp_name = osp.basename(args.checkpoint)
            video_fname = f"{args.trajectory.split('.csv')[0]}_{cp_name.split('.zip')[0]}.mp4"
            video_fpath = osp.join(video_output, 'ppo', video_fname)
            print(f'Recording to {video_fpath}')
            model.demo(video_file=video_fpath)
        model.demo()
    except Exception as e:
        print(e)