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
    parser.add_argument('-w', '--waypoints', help='Text file with waypoints.', default='data/waypoints/wps15.txt')
    parser.add_argument('-m', '--model', help='Model directory.', default='models/mevea/mantsinen/ppo')
    parser.add_argument('-c', '--checkpoint', help='Checkpoint', default='first', choices=['first', 'last', 'best'])
    parser.add_argument('-v', '--video', help='Record video?', type=bool, default=False)
    args = parser.parse_args()

    chkpt_dir = args.model

    # check that server is running

    while not is_server_running(server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # extract waypoints

    waypoints = get_test_waypoints(args.waypoints)

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
        wp_size,
        tstep,
        bonus
    )]
    env = MeveaVecEnv(env_fns)

    # load model and run it in demo mode

    try:
        model = ppo(policy, env, 1, policy_kwargs=dict(net_arch=ppo_net_arch), batch_size=batch_size, n_steps=nsteps,
                    model_path=chkpt_dir, chkpt_name=args.checkpoint, tensorboard_log='tensorboard_log', verbose=1)
        cp_name = osp.basename(args.checkpoint)
        wp_name = osp.basename(args.waypoints)
        img_fname = f"{wp_name.split('.txt')[0]}_{cp_name}.png"
        img_fpath = osp.join(img_output, 'ppo', img_fname)
        if args.video:
            video_fname = f"{wp_name.split('.txt')[0]}_{cp_name}.mp4"
            video_fpath = osp.join(video_output, 'ppo', video_fname)
            print(f'Recording to {video_fpath}')
        else:
            video_fpath = None
        model.demo(img_file=img_fpath, video_file=video_fpath)
    except Exception as e:
        print(e)