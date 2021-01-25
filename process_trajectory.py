import argparse, os
import os.path as osp
import numpy as np

from common.data_utils import load_signals, save_trajectory
from common.solver_utils import get_solver_path, start_solver, stop_solver
from common.server_utils import is_server_running, is_backend_registered, post_signals, get_state

from time import sleep

if __name__ == '__main__':

    # parameters

    output_dir = 'data/trajectory_examples'
    signal_csv = {
        'input': 'data/signals/input.csv',
        'output': 'data/signals/output.csv',
        'reward': 'data/signals/reward.csv'
    }
    sleep_interval = 3

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-o', '--output', help='Output file.', default='trajectory4.csv')
    parser.add_argument('-d', '--distance', help='Minimal distance.', default=0.001, type=float)
    parser.add_argument('-t', '--time', help='Maximal time.', default=1, type=float)
    args = parser.parse_args()

    # check whether such file already exists

    fnames = os.listdir(output_dir)
    output_file = args.output
    if args.output in fnames:
        print('Such trajectory file already exists.')
        go = input('Do you want to enter new name (y/n)?\n')
        if go == 'y':
            output_file = input('Enter new output file name:\n')
    output = osp.join(output_dir, output_file)

    # check that server is running

    while not is_server_running(args.server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # start one solver

    solver_path = get_solver_path()
    proc = start_solver(solver_path, args.model)
    backend_id = proc.pid

    # wait for backend to be registered

    while not is_backend_registered(args.server, backend_id):
        print('Mevea Solver is not running, please wait...')
        sleep(sleep_interval)
    print('\nMevea Solver is up and running!')

    # add signals

    signals = {}
    for key in signal_csv.keys():
        values, _, _ = load_signals(signal_csv[key])
        signals[key] = values
    result = post_signals(args.server, backend_id, signals)

    # print message

    print('\nIn solver, go to Control -> Input control -> Start playback, and select one of the pre-recorded inputs from data/input_trajectories\n')

    # wait for the machine to start moving

    states = []
    rewards = []
    timestamps = []
    moving = False
    last_reward = []
    last_t_state = None
    move_t_start = None
    while not moving:
        state, reward, t_state = get_state(args.server, backend_id)
        if len(state) > 0 and len(reward) > 0 and t_state is not None:
            if len(last_reward) > 0 and last_t_state is not None:
                if t_state > last_t_state:
                    dist = np.linalg.norm(np.array(reward) - np.array(last_reward))
                    if dist > args.distance:
                        if move_t_start is None:
                            move_t_start = last_t_state
                        elif t_state > move_t_start + args.time:
                            moving = True
                    else:
                        move_t_start = None
                    last_reward = reward
                    last_t_state = t_state
                    states.append(state)
                    rewards.append(reward)
                    timestamps.append(t_state)
                    if timestamps[0] < t_state - args.time:
                        del timestamps[0]
                        del states[0]
                        del rewards[0]
            else:
                last_reward = reward
                last_t_state = t_state

    print('Recording the trajectory...')

    # record the trajectory and wait for the moving to stop

    stay_t_start = None
    while moving:
        state, reward, t_state = get_state(args.server, backend_id)
        if state is not None and reward is not None and t_state is not None:
            if t_state > last_t_state:
                dist = np.linalg.norm(np.array(reward) - np.array(last_reward))
                if dist <= args.distance:
                    if stay_t_start is None:
                        stay_t_start = last_t_state
                    elif t_state > stay_t_start + args.time:
                        moving = False
                else:
                    stay_t_start = None
                last_reward = reward
                last_t_state = t_state
                states.append(state)
                rewards.append(reward)
                timestamps.append(t_state)

    # save trajectory

    save_trajectory(signals, timestamps, rewards, states, stay_t_start, output)
    print('Trajectory has been recorded and saved in {0}!'.format(output))

    # stop the solver

    stop_solver(proc)
    print('\nMevea Solver has stopped!')