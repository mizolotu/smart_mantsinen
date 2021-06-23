import sys, requests, os
from time import sleep

# gains and thresholds

attempts_max = 10  # maximum number of attempts to register
exec_freq = 1000  # Hz
sleep_interval = 3  # seconds

# http parameters

http_url = 'http://127.0.0.1:5000'
id_uri = 'id'
state_uri = 'post_state'
action_uri = 'get_action'
signals_uri = 'signals'

def post_id(id):
    uri = '{0}/{1}'.format(http_url, id_uri)
    try:
        r = requests.post(uri, json={'id': id})
        jdata = r.json()
        config_time = jdata['t_config']
    except Exception:
        config_time = None
    return config_time

def get_signals(id):
    uri = '{0}/{1}'.format(http_url, signals_uri)
    try:
        r = requests.get(uri, json={'id': id})
        jdata = r.json()
        signals = jdata['signals']
        config_time = jdata['t_config']
    except:
        signals = {}
        config_time = None
    return signals, config_time

def post_state_and_reward(id, state, reward, conditional, t):
    uri = '{0}/{1}'.format(http_url, state_uri)
    r_time = None
    try:
        r = requests.post(uri, json={'id': id, 'state': state, 'reward': reward, 'conditional': conditional, 't_state_simulation': t})
        r_time = r.json()['t_state_real']
    except Exception:
        print(Exception)
    return r_time

def get_action(id):
    uri = '{0}/{1}'.format(http_url, action_uri)
    try:
        r = requests.get(uri, json={'id': id})
        jdata = r.json()
        action = jdata['action']
        conditional_input = jdata['conditional']
        action_submission_real_time = jdata['t_action_real']
        action_duration_simulation_time = jdata['t_action_simulation']
    except:
        action = []
        conditional_input = []
        action_submission_real_time = None
        action_duration_simulation_time = None
    return action, conditional_input, action_submission_real_time, action_duration_simulation_time

def initScript():

    # register

    backend_id = os.getpid()
    id_time = None
    n_attempts = 0
    print('Trying to register...')
    while id_time is None:
        id_time = post_id(backend_id)
        sleep(sleep_interval)
        n_attempts += 1
        if n_attempts >= attempts_max:
            sys.exit(1)
    print('Successfully registerd with id {0}!'.format(backend_id))
    GObject.data['id'] = backend_id

    # init signals

    GObject.data['signals'] = None

    # init action variables

    GObject.data['current_action'] = None
    GObject.data['current_conditional'] = None

    # init time moments

    GObject.data['last_iteration_simulation_time'] = 0
    GObject.data['last_action_real_time'] = 0
    GObject.data['current_action_end_simulation_time'] = 0
    GObject.data['last_state_sent_simulation_time'] = 0

    # other

    GObject.data['in_action'] = False

def callScript(deltaTime, simulationTime):

    # check if required time passed

    time_passed = False
    t_now = simulationTime
    t_last = GObject.data['last_iteration_simulation_time']
    t_delta = t_now - t_last
    if t_delta > 1.0 / exec_freq:
        time_passed = True

    # report state and reward if the required time passed

    if time_passed:

        # check if we have signals

        if GObject.data['signals'] is not None:

            # query an action

            print('querying new action')
            action, conditional_input, action_submission_real_time, action_duration_simulation_time = get_action(GObject.data['id'])

            # what if this is a new action

            if action_submission_real_time is not None and action_submission_real_time > GObject.data['last_action_real_time']:

                # save the action

                GObject.data['current_action'] = action
                GObject.data['current_conditional'] = conditional_input
                GObject.data['last_action_real_time'] = action_submission_real_time
                GObject.data['current_action_end_simulation_time'] = simulationTime + action_duration_simulation_time
                GObject.data['in_action'] = True

                # report the result

                state_values = [x.value() for x in GObject.data['state_objects']]
                reward_values = [x.value() for x in GObject.data['reward_objects']]
                conditional_values = [x.value() for x in GObject.data['conditional_objects']]
                post_state_and_reward(GObject.data['id'], state_values, reward_values, conditional_values, t_now)
                GObject.data['last_state_sent_simulation_time'] = simulationTime

            # check if the action is real and it still can be executed

            if simulationTime <= GObject.data['current_action_end_simulation_time']:

                if action is not None:

                    print(f"executing action {GObject.data['current_action']} for {GObject.data['current_action_end_simulation_time'] - simulationTime} seconds")

                    # all good, execute the action

                    assert len(GObject.data['current_action']) == len(GObject.data['signals']['input'])
                    for i, input_signal in enumerate(GObject.data['signals']['input']):
                        GDict[input_signal].setInputValue(GObject.data['current_action'][i])

                    assert len(GObject.data['current_conditional']) == len(GObject.data['signals']['conditional'])
                    for i, input_signal in enumerate(GObject.data['signals']['conditional']):
                        GDict[input_signal].setInputValue(GObject.data['current_conditional'][i])

            else:

                # keep reporting the result

                state_values = [x.value() for x in GObject.data['state_objects']]
                reward_values = [x.value() for x in GObject.data['reward_objects']]
                conditional_values = [x.value() for x in GObject.data['conditional_objects']]
                post_state_and_reward(GObject.data['id'], state_values, reward_values, conditional_values, t_now)
                GObject.data['last_state_sent_simulation_time'] = simulationTime

        # get signals if they are still None

        else:

            signals, signals_time = get_signals(GObject.data['id'])
            if signals != {}:
                GObject.data['signals'] = signals

                # save component names

                GObject.data['state_objects'] = [
                    GSolver.getParameter(input_signal, 'InputValue') for input_signal in GObject.data['signals']['input']
                ] + [
                    GSolver.getParameter(output_signal, 'value') for output_signal in GObject.data['signals']['output'] + GObject.data['signals']['reward']
                ]
                GObject.data['reward_objects'] = [
                    GSolver.getParameter(output_signal, 'value') for output_signal in GObject.data['signals']['reward']
                ]
                GObject.data['conditional_objects'] = [
                    GSolver.getParameter(item, 'InputValue') for item in GObject.data['signals']['conditional']
                ]

                # post the very first observation

                state_values = [x.value() for x in GObject.data['state_objects']]
                reward_values = [x.value() for x in GObject.data['reward_objects']]
                conditional_values = [x.value() for x in GObject.data['conditional_objects']]
                post_state_and_reward(GObject.data['id'], state_values, reward_values, conditional_values, t_now)
                GObject.data['last_state_sent_simulation_time'] = simulationTime

        # save the last iteration time

        GObject.data['last_iteration_simulation_time'] = simulationTime