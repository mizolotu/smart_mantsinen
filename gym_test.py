import os, sys
import numpy as np

from common import logger
from gym.envs.classic_control.pendulum import PendulumEnv
from common.gym_vec_env import SubprocVecEnv
from common.policies import MlpPolicy
from common.runners import Runner
from common.callbacks import CheckpointCallback
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.model_utils import find_checkpoint_with_max_step

def make_env(env_class):
    fn = lambda: env_class()
    return fn

def generate_traj(env, model, nsteps):
    n = len(env.remotes)
    states = [[] for _ in range(n)]
    actions = [[] for _ in range(n)]
    obs = env.reset()
    for i in range(nsteps):
        action, state = model.predict(obs)
        next_obs, reward, done, info = env.step(action)
        for e in range(n):
            states[e].append(obs[e])
            actions[e].append(action[e])
        obs = np.array(next_obs)
    return states, actions

if __name__ == '__main__':

    nenvs = 4
    nsteps = 10000

    logdir = 'models/gym/pendulum/ppo/'
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    env_fns = [make_env(PendulumEnv) for _ in range(nenvs)]
    env = SubprocVecEnv(env_fns)
    print(env.action_space.shape)

    try:
        checkpoint_file = find_checkpoint_with_max_step('{0}/model_checkpoints/'.format(logdir))
        model = ppo.load(checkpoint_file)
        model.set_env(env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))
    except Exception as e:
        print(e)
        model = ppo(MlpPolicy, env, runner=Runner, verbose=1)
        model.learn(total_timesteps=nsteps)
        model.save('{0}/model_checkpoints/rl_model_0_steps.zip'.format(logdir))

    trajs = []
    for i in range(1):
        states, actions = generate_traj(env, model, 2048)
        for se, ae in zip(states, actions):
            trajs.append([])
            for s, a in zip(se, ae):
                trajs[-1].append(np.hstack([s, a]))
            trajs[-1] = np.vstack(trajs[-1])

    del model

    model = ppo(MlpPolicy, env, runner=Runner, verbose=1)
    model.pretrain(trajs, n_epochs=100)
    model.learn(total_timesteps=nsteps)

    #checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='{0}/model_checkpoints/'.format(logdir))
    #model.learn(total_timesteps=nsteps, callback=[checkpoint_callback])
