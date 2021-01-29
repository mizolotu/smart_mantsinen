import argparse, os
import os.path as osp
import numpy as np

from common.data_utils import load_signals, save_trajectory, parse_conditional_signals, is_moving
from common.solver_utils import get_solver_path, start_solver, stop_solver
from common.server_utils import is_server_running, is_backend_registered, post_signals, get_state

from time import sleep

if __name__ == '__main__':

    # parameters

    output_dir = 'data/trajectory_examples'
    signal_dir = 'data/signals'
    sleep_interval = 3

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-o', '--output', help='Output file.', default='trajectory3.csv')
    parser.add_argument('-d', '--debug', help='Debug.', default=False, type=bool)
    args = parser.parse_args()

    # check whether such file already exists

    fnames = os.listdir(output_dir)
    output_file = args.output
    if args.output in fnames:
        print('Trajectory file with name {0} already exists.'.format(args.output))
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
    if not args.debug:
        proc = start_solver(solver_path, args.model)
        backend_id = proc.pid
    else:
        backend_id = input('Start Mevea solver manually, and input the process id:\n')
        backend_id = int(backend_id)

    # wait for backend to be registered

    while not is_backend_registered(args.server, backend_id):
        print('Mevea Solver is not running, please wait...')
        sleep(sleep_interval)

    # add signals

    signals = {}
    for key in ['input', 'output', 'reward', 'conditional']:
        values = load_signals(signal_dir, key)
        signals[key] = values[0]
        if key == 'conditional':
            conditional_signals = parse_conditional_signals(values, signals['input'])
    result = post_signals(args.server, backend_id, signals)

    # print message

    print('\nMevea Solver is about to start. Once it has started, go to Control -> Input control -> Start playback, and select one of the pre-recorded inputs from data/input_trajectories\n')

    # wait for the machine to start moving

    states = []
    rewards = []
    timestamps = []
    moving = False
    last_t_state = None
    move_t_start = None
    while not moving:
        state, reward, conditional_values, t_state = get_state(args.server, backend_id)
        moving = is_moving(conditional_signals, conditional_values, t_state)
        last_t_state = t_state
        if moving:
            timestamps.append(t_state)
            states.append(state)
            rewards.append(reward)

    print('Recording the trajectory...')

    # record the trajectory and wait for the moving to stop

    while moving:
        state, reward, conditional_values, t_state = get_state(args.server, backend_id)
        if state is not None and reward is not None and t_state is not None:
            if t_state > last_t_state:
                moving = is_moving(conditional_signals, conditional_values, t_state)
                last_t_state = t_state
                if moving:
                    states.append(state)
                    rewards.append(reward)
                    timestamps.append(t_state)

    # save trajectory

    save_trajectory(signals, timestamps, rewards, states, output)
    print('Trajectory has been recorded and saved in {0}!'.format(output))

    # stop the solver

    stop_solver(proc)
    print('\nMevea Solver has stopped!')