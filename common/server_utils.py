import requests

def is_server_running(url, timeout=5):
    result = False
    try:
        requests.get(url, timeout=timeout)
        result = True
    except Exception as e:
        print(e)
    return result

def is_backend_registered(url, id):
    result = False
    try:
        r = requests.get('{0}/id'.format(url), json={'id': id})
        if r.json()['t_config'] is not None:
            result = True
    except Exception as e:
        print(e)
    return result

def post_signals(url, id, signals):
    result = False
    try:
        r = requests.post('{0}/signals'.format(url), json={'id': id, 'signals': signals})
        data = r.json()
        if 't_config' in data.keys() and data['t_config'] is not None:
            result = True
    except Exception as e:
        print(e)
    return result

def get_state(url, id, last_state_time):
    state = []
    reward = []
    conditional = []
    t_state_real = None
    t_state_simulation = None
    ready = False
    while not ready:
        try:
            print(f'Env {id} in get_state')
            r = requests.get('{0}/get_state'.format(url), json={'id': id})
            data = r.json()
            if 't_state_simulation' in data.keys() and data['t_state_simulation'] is not None:
                if last_state_time is None or data['t_state_simulation'] >= last_state_time:
                    if 'state' in data.keys() and 'reward' in data.keys() and 't_state_real' in data.keys() and 'conditional' in data.keys():
                        state = data['state']
                        reward = data['reward']
                        conditional = data['conditional']
                        t_state_real = data['t_state_real']
                        t_state_simulation = data['t_state_simulation']
                        ready = True
                #if last_state_time is not None and data['t_state_simulation'] <= last_state_time:
                #    print(f"The solver {id} is ahead, have to wait for {last_state_time - data['t_state_simulation']} seconds.")
        except Exception as e:
            print(e)
    return state, reward, conditional, t_state_real, t_state_simulation

def post_action(url, id, action, conditional, next_simulation_time):
    ready = False
    while not ready:
        try:
            print(f'Env {id} in post_action')
            r = requests.post('{0}/post_action'.format(url), json={'id': id, 'action': action, 'conditional': conditional, 't_action_simulation': next_simulation_time})
            data = r.json()
            if 't_action_real' in data.keys() and data['t_action_real'] is not None:
                ready = True
        except Exception as e:
            print(e)

def delete_id(url, id):
    result = False
    try:
        r = requests.delete('{0}/id'.format(url), json={'id': id})
        if r.json()['t_config'] is not None:
            result = True
    except Exception as e:
        print(e)
    return result