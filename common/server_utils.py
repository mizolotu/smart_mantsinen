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

def get_state(url, id):
    state = []
    reward = []
    conditional = []
    t_state = None
    try:
        r = requests.get('{0}/state'.format(url), json={'id': id})
        data = r.json()
        if 'state' in data.keys() and 'reward' in data.keys() and 't_state' in data.keys() and 'conditional' in data.keys():
            state = data['state']
            reward = data['reward']
            conditional = data['conditional']
            t_state = data['t_state']
    except Exception as e:
        print(e)
    return state, reward, conditional, t_state

def set_action(url, id, action, conditional):
    result = False
    try:
        r = requests.post('{0}/action'.format(url), json={'id': id, 'action': action, 'conditional': conditional})
        data = r.json()
        if 't_config' in data.keys() and data['t_config'] is not None:
            result = True
    except Exception as e:
        print(e)
    return result

def delete_id(url, id):
    result = False
    try:
        r = requests.delete('{0}/id'.format(url), json={'id': id})
        if r.json()['t_config'] is not None:
            result = True
    except Exception as e:
        print(e)
    return result