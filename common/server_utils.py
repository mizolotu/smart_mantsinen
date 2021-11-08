import requests, os, sys, psutil

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

def get_state(url, id, last_state_time, is_last_step=False, step_count=None, pid_check_freq=10):
    state = []
    reward = []
    conditional = []
    t_state_real = None
    t_state_simulation = None
    ready = False
    open('log_state.txt', 'w').close()
    lines = []
    attempts = 0
    crashed = False
    while not ready:
        try:
            r = requests.get('{0}/get_state'.format(url), json={'id': id})
            data = r.json()
            lines.append(f"{id}, 1, {'t_state_simulation' in data.keys()}, {last_state_time}\n")
            if 't_state_simulation' in data.keys() and data['t_state_simulation'] is not None:
                lines.append(f"{id}, 2, {last_state_time is None}, {data['t_state_simulation']}, {last_state_time}, {is_last_step}, {step_count}\n")
                attempts += 1
                if attempts % pid_check_freq == 0:
                    if id not in (p.pid for p in psutil.process_iter()):
                        crashed = True
                        break
                if last_state_time is None or data['t_state_simulation'] >= last_state_time or is_last_step:
                    lines.append(f"{id} 3, {'state' in data.keys() and 'reward' in data.keys() and 't_state_real' in data.keys() and 'conditional' in data.keys()}\n")
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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(e, fname, exc_tb.tb_lineno)
        if len(lines) > 100:
            with open('log_state.txt', 'a') as f:
                f.writelines(lines)
            lines = []
    return state, reward, conditional, t_state_real, t_state_simulation, crashed

def post_action(url, id, action, conditional, next_simulation_time):
    ready = False
    lines = []
    open('log_action.txt', 'w').close()
    while not ready:
        try:
            r = requests.post('{0}/post_action'.format(url), json={'id': id, 'action': action, 'conditional': conditional, 't_action_simulation': next_simulation_time})
            data = r.json()
            lines.append(f"{'t_action_real' in data.keys()} {data['t_action_real'] is not None}\n")
            if 't_action_real' in data.keys() and data['t_action_real'] is not None:
                ready = True
        except Exception as e:
            print(e)
        if len(lines) > 100:
            with open('log_action.txt', 'a') as f:
                f.writelines(lines)
            lines = []

def delete_id(url, id):
    result = False
    try:
        r = requests.delete('{0}/id'.format(url), json={'id': id})
        if r.json()['t_config'] is not None:
            result = True
    except Exception as e:
        print(e)
    return result