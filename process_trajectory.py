import argparse, os
import os.path as osp
import numpy as np

from common.data_utils import load_signals, save_trajectory, parse_conditional_signals, is_moving, is_acting
from common.solver_utils import get_solver_path, start_solver, stop_solver
from common.server_utils import is_server_running, is_backend_registered, post_signals, post_action, get_state
from config import default_actions, trajectory_dir, signal_dir

from collections import deque
from time import sleep

if __name__ == '__main__':

    # parameters

    sleep_interval = 3

    # process arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='File path to the model.', default='C:\\Users\\mevea\\MeveaModels\\Mantsinen\\Models\\Mantsinen300M\\300M_fixed.mvs')
    parser.add_argument('-s', '--server', help='Server URL.', default='http://127.0.0.1:5000')
    parser.add_argument('-o', '--output', help='Output file.', default='trajectory1.csv')
    parser.add_argument('-d', '--debug', help='Debug.', default=False, type=bool)
    args = parser.parse_args()

    # check whether such file already exists

    fnames = os.listdir(trajectory_dir)
    output_file = args.output
    if args.output in fnames:
        print('Trajectory file with name {0} already exists.'.format(args.output))
        go = input('Do you want to enter new name (y/n)?\n')
        if go == 'y':
            output_file = input('Enter new output file name:\n')
    output = osp.join(trajectory_dir, output_file)

    # check that server is running

    while not is_server_running(args.server):
        print('Start the server: python3 env_server.py')
        sleep(sleep_interval)

    # start one solver

    solver_path = get_solver_path()
    if not args.debug:
        proc = start_solver(solver_path, args.model, headless=False)
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
    is_acting_list = []
    conditional_values = []
    timestamps = []
    moving, acting = False, False
    last_t_state, move_t_start = None, None

    last_states = deque(maxlen=1000)

    while not (moving and acting):
        post_action(args.server, backend_id, None, conditional_values, next_simulation_time=0)
        state, reward, conditional_values, t_state_real, t_state_simulation, _ = get_state(args.server, backend_id, last_state_time=0)
        moving = is_moving(conditional_signals, conditional_values, t_state_simulation)
        acting = is_acting(signals, state, default_actions)
        if moving and acting:
            timestamps.append(t_state_simulation)
            states.append(state)
            rewards.append(reward)
            is_acting_list.append(True)
            print(state)
            for ls in last_states:
                print(ls)
        last_time = t_state_simulation
        last_states.append(state)
        last_reward = reward

    print('Recording the trajectory...')

    # record the trajectory and wait for the moving to stop

    moving = True
    while moving:
        post_action(args.server, backend_id, None, conditional_values, next_simulation_time=0)
        state, reward, conditional_values, t_state_real, t_state_simulation, _ = get_state(args.server, backend_id, last_state_time=0)
        if state is not None and reward is not None and t_state_simulation is not None:
            moving = is_moving(conditional_signals, conditional_values, t_state_simulation)
            if moving:
                states.append(state)
                rewards.append(reward)
                timestamps.append(t_state_simulation)
                is_acting_list.append(is_acting(signals, state, default_actions))

    # last acting

    is_acting_array = np.array(is_acting_list)
    idx_last = np.where(is_acting_array == True)[0][-1]

    # save trajectory

    save_trajectory(signals, timestamps[:idx_last], rewards[:idx_last], states[:idx_last], output[:idx_last])
    print('Trajectory has been recorded and saved in {0}!'.format(output))

    # stop the solver

    stop_solver(proc)
    print('\nMevea Solver has stopped!')