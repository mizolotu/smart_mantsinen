import logging, json

from flask import Flask, jsonify, request
from time import time, sleep
from threading import Thread

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True

@app.route('/')
def hello():
    return 'Server is running!'

@app.route('/backends')
def get_backends():
    global backends
    return jsonify(backends)

@app.route('/id', methods=['GET', 'POST', 'DELETE'])
def get_post_or_delete_id():
    global backend_ids, backends
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        t_config = None
        if request.method == 'GET':
            if id in backend_ids:
                idx = backend_ids.index(id)
                if 't_config' in backends[idx].keys():
                    t_config = backends[backend_ids.index(id)]['t_config']
        elif request.method == 'POST':
            if id not in backend_ids:
                backend_ids.append(id)
                t_config = time()
                backends.append({'t_config': t_config})
        elif request.method == 'DELETE':
            if id in backend_ids:
                idx = backend_ids.index(id)
                t_config = backends[idx]['t_config']
                del backend_ids[idx]
                del backends[idx]
        result = {'t_config': t_config}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/signals', methods=['GET', 'POST', 'DELETE'])
def get_post_or_delete_signals():
    global backend_ids, backends
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            t_config = backends[idx]['t_config']
            signals = {}
            if request.method == 'GET':
                if 'signals' in backends[idx].keys():
                    for key in signal_keys:
                        if key in backends[idx]['signals'].keys():
                            signals.update({key: backends[idx]['signals'][key]})
                result = {'signals': signals, 't_config': t_config}
            elif request.method == 'POST':
                if 'signals' in jdata.keys():
                    signals = {}
                    for key in signal_keys:
                        if key in jdata['signals'].keys():
                            signals[key] = jdata['signals'][key]
                    backends[idx]['signals'] = signals
                    t_config = time()
                    backends[idx].update({'t_config': t_config})
                    result = {'t_config': t_config}
                else:
                    result = {'Error': 'wrong json format'}
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/get_state', methods=['GET'])
def get_state_and_reward():
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            result = {'t_state_real': None, 't_state_simulation': None, 'state': [], 'reward': [], 'conditional': []}
            for key in result.keys():
                if key in backends[idx].keys():
                    value = backends[idx][key]
                    result.update({key: value})
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/post_state', methods=['POST'])
def post_state_and_reward():
    global backend_ids, backends
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            backends[idx].update({'t_state_real': time()})
            for key in ['state', 'reward', 'conditional', 't_state_simulation']:
                if key in jdata.keys():
                    value = jdata[key]
                    backends[idx].update({key: value})
            result = {'t_state_real': backends[idx]['t_state_real']}
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/get_action', methods=['GET'])
def get_action():
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            result = {'t_action_real': None, 't_action_simulation': None, 'action': [], 'conditional': []}
            for key in result.keys():
                if key in backends[idx].keys():
                    value = backends[idx][key]
                    result.update({key: value})
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/post_action', methods=['POST'])
def get_or_post_action():
    global backend_ids, backends
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            backends[idx].update({'t_action_real': time()})
            for key in ['action', 'conditional', 't_action_simulation']:
                if key in jdata.keys():
                    value = jdata[key]
                    backends[idx].update({key: value})
            result = {'t_action_real': backends[idx]['t_action_real']}
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

def _remove_idle_backends(timeout=120):
    global backend_ids, backends
    while True:
        t_now = time()
        for idx, backend in enumerate(backends):
            t_last = backend['t_config']
            for key in ['t_state_real', 't_action_real']:
                if key in backend.keys() and backend[key] > t_last:
                    t_last = backend[key]
            if t_last < t_now - timeout:
                del backend_ids[idx]
                del backends[idx]
        sleep(timeout)

if __name__ == '__main__':
    signal_keys = ['input', 'output', 'reward', 'conditional']
    backend_ids = []
    backends = []
    rib_th = Thread(target=_remove_idle_backends, daemon=True)
    rib_th.start()
    app.run(host='0.0.0.0', threaded=True)