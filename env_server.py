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
    global backends
    backend_ids = [backend['id'] for backend in backends]
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        t_config = None
        if request.method == 'GET':
            if id in backend_ids and 't_config' in backends[backend_ids.index(id)].keys():
                t_config = backends[backend_ids.index(id)]['t_config']
        elif request.method == 'POST':
            if id not in backend_ids:
                t_config = time()
                backends.append({'id': id, 't_config': t_config})
        elif request.method == 'DELETE':
            if id in backend_ids:
                idx = backend_ids.index(id)
                t_config = backends[idx]['t_config']
                del backends[idx]
        result = {'t_config': t_config}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/signals', methods=['GET', 'POST', 'DELETE'])
def get_post_or_delete_signals():
    global backends
    backend_ids = [backend['id'] for backend in backends]
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            signals = {}
            if request.method == 'GET' and 'signals' in backends[idx].keys():
                for key in ['input', 'output', 'reward']:
                    if key in backends[idx]['signals'].keys():
                        signals.update({key: backends[idx]['signals'][key]})
                result = {'signals': signals}
            elif request.method == 'POST':
                if 'signals' in jdata.keys():
                    t_config = time()
                    backends[idx].update({'t_config': t_config})
                    signals = {}
                    for key in ['input', 'output', 'reward']:
                        if key in jdata['signals'].keys():
                            signals[key] = jdata['signals'][key]
                    backends[idx]['signals'] = signals
                    result = {'t_config': t_config}
                else:
                    result = {'Error': 'wrong json format'}
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/state', methods=['GET', 'POST'])
def get_or_post_state_and_reward():
    global backends
    backend_ids = [backend['id'] for backend in backends]
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            if request.method == 'GET':
                result = {'t_state': None, 'state': [], 'reward': []}
                for key in result.keys():
                    if key in backends[idx].keys():
                        value = backends[idx][key]
                        result.update({key: value})
            elif request.method == 'POST':
                backends[idx].update({'t_state': time()})
                for key in ['state', 'reward']:
                    if key in jdata.keys():
                        value = jdata[key]
                        backends[idx].update({key: value})
                result = {'t_config': backends[idx]['t_config']}
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

@app.route('/action', methods=['GET', 'POST'])
def get_or_post_action():
    global backends
    backend_ids = [backend['id'] for backend in backends]
    data = request.data.decode('utf-8')
    try:
        jdata = json.loads(data)
    except:
        jdata = {}
    if 'id' in jdata.keys():
        id = jdata['id']
        if id in backend_ids:
            idx = backend_ids.index(id)
            if request.method == 'GET':
                result = {'t_action': None, 'action': []}
                for key in result.keys():
                    if key in backends[idx].keys():
                        value = backends[idx][key]
                        result.update({key: value})
            elif request.method == 'POST':
                backends[idx].update({'t_action': time()})
                for key in ['action']:
                    if key in jdata.keys():
                        value = jdata[key]
                        backends[idx].update({key: value})
                result = {'t_config': backends[idx]['t_config']}
        else:
            result = {'Error': 'no such backend'}
    else:
        result = {'Error': 'wrong json format'}
    return jsonify(result)

def _remove_idle_backends(timeout=30):
    global backends
    while True:
        t_now = time()
        for idx, backend in enumerate(backends):
            t_last = backend['t_config']
            for key in ['t_state', 't_action']:
                if key in backend.keys() and backend[key] > t_last:
                    t_last = backend[key]
            if t_last < t_now - timeout:
                del backends[idx]
        sleep(timeout)

if __name__ == '__main__':
    backends = []
    rib_th = Thread(target=_remove_idle_backends, daemon=True)
    rib_th.start()
    app.run(host='0.0.0.0')