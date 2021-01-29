import sys, requests, os
from time import time, sleep

# gains and thresholds

attempts_max = 10  # maximum number of attempts to register
exec_freq = 1000  # Hz
sleep_interval = 1  # seconds

# http parameters

http_url = 'http://127.0.0.1:5000'
id_uri = 'id'
state_uri = 'state'
action_uri = 'action'
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
    except:
        signals = {}
    return signals

def post_state_and_reward(id, state, reward, conditional):
    uri = '{0}/{1}'.format(http_url, state_uri)
    config_time = None
    try:
        r = requests.post(uri, json={'id': id, 'state': state, 'reward': reward, 'conditional': conditional})
        config_time = r.json()['t_config']
    except Exception:
        print(Exception)
    return config_time

def get_action(id):
    uri = '{0}/{1}'.format(http_url, action_uri)
    try:
        r = requests.get(uri, json={'id': id})
        jdata = r.json()
        action = jdata['action']
        conditional_input = jdata['conditional']
        action_time = jdata['t_action']
    except:
        action = []
        conditional_input = []
        action_time = None
    return action, conditional_input, action_time

def initScript():

    # try to register

    backend_id = os.getpid()
    config_time = None
    n_attempts = 0
    print('Trying to register...')
    while config_time is None:
        config_time = post_id(backend_id)
        sleep(sleep_interval)
        n_attempts += 1
        if n_attempts >= attempts_max:
            sys.exit(1)
    print('Successfully registerd with id {0}!'.format(backend_id))
    GObject.data['id'] = backend_id
    GObject.data['last_config_time'] = config_time

    # signals

    GObject.data['signals'] = {
        'input': [],
        'output': [],
        'reward': [],
        'conditional': []
    }

    # components

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

    # times

    GObject.data['last_iteration_time'] = time()
    GObject.data['last_action_time'] = time()

def callScript(deltaTime, simulationTime):

    # check if required time passed

    time_passed = False
    t_now = time()
    t_last = GObject.data['last_iteration_time']
    t_delta = t_now - t_last
    if t_delta > 1.0 / exec_freq:
        time_passed = True

    # report state and reward if the required time passed

    if time_passed:

        # execute new action

        action, conditional_input, action_time = get_action(GObject.data['id'])
        if action_time is not None and action_time > GObject.data['last_action_time']:

            assert len(action) == len(GObject.data['signals']['input'])
            for i, input_signal in enumerate(GObject.data['signals']['input']):
                GDict[input_signal].setInputValue(action[i])

            assert len(conditional_input) == len(GObject.data['signals']['conditional'])
            for i, input_signal in enumerate(GObject.data['signals']['conditional']):
                GDict[input_signal].setInputValue(conditional_input[i])

            GObject.data['last_action_time'] = action_time

        # report the result

        state_values = [x.value() for x in GObject.data['state_objects']]
        reward_values = [x.value() for x in GObject.data['reward_objects']]
        conditional_values = [x.value() for x in GObject.data['conditional_objects']]
        config_time = post_state_and_reward(GObject.data['id'], state_values, reward_values, conditional_values)
        if config_time is None:
            print('Looks like server is not running, or backend {0} is not registered!'.format(GObject.data['id']))

        # save the last iteration time

        GObject.data['last_iteration_time'] = time()

        # get signals if config time is updated

        if config_time > GObject.data['last_config_time']:

            signals = get_signals(GObject.data['id'])

            GObject.data['signals'].update(signals)

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