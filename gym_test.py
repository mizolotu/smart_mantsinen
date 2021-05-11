import os, sys, pandas
import numpy as np

from common import logger
from gym.envs.classic_control.pendulum import PendulumEnv
from common.gym_vec_env import SubprocVecEnv
from common.policies import MlpPolicy as ppo_policy
from baselines.ddpg.policies import MlpPolicy as ddpg_policy
from common.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from common.runners import Runner
from common.callbacks import CheckpointCallback
from baselines.ppo2.ppo2 import PPO2 as ppo
from baselines.ddpg.ddpg import DDPG as ddpg
from common.model_utils import find_checkpoint_with_max_step

def make_env(env_class):
    fn = lambda: env_class()
    return fn

def generate_traj(env, model, nsteps):
    n = len(env.remotes)
    states = [[] for _ in range(n)]
    actions = [[] for _ in range(n)]
    next_states = [[] for _ in range(n)]
    rewards = [[] for _ in range(n)]
    obs = env.reset()
    for i in range(nsteps):
        action, state = model.predict(obs)
        next_obs, reward, done, info = env.step(action)
        for e in range(n):
            states[e].append(obs[e])
            actions[e].append(action[e])
            next_states[e].append(next_obs[e])
            rewards[e].append(reward[e])
        obs = np.array(next_obs)
    return states, actions, next_states, rewards

if __name__ == '__main__':

    nenvs = 4
    nsteps = nenvs * int(1e6)
    trajs_fpath = 'data/gymtest/trajs.csv'
    trainer_model_dir = 'models/gym/pendulum/ppo'
    student_model_dir = 'models/gym/pendulum/ddpg'

    trainer_env_fns = [make_env(PendulumEnv) for _ in range(nenvs)]
    trainer_env = SubprocVecEnv(trainer_env_fns)

    student_env_fns = [make_env(PendulumEnv) for _ in range(1)]
    student_env = SubprocVecEnv(student_env_fns)

    eval_env_fns = [make_env(PendulumEnv) for _ in range(1)]
    eval_env = SubprocVecEnv(eval_env_fns)

    try:
        checkpoint_file = find_checkpoint_with_max_step('{0}/model_checkpoints/'.format(trainer_model_dir))
        trainer_model = ppo.load(checkpoint_file)
        trainer_model.set_env(trainer_env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))
    except Exception as e:
        print(e)
        #model = ddpg(
        #    MlpPolicy,
        #    env,
        #    eval_env=eval_env,
        #    verbose=1,
        #    param_noise=AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1),
        #    action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.5 * np.ones(env.action_space.shape[0]))
        #)
        trainer_model = ppo(ppo_policy, trainer_env, runner=Runner, eval_env=eval_env)
        trainer_model.learn(total_timesteps=nsteps)
        trainer_model.save('{0}/model_checkpoints/rl_model_0_steps.zip'.format(trainer_model_dir))

    try:
        p = pandas.read_csv(trajs_fpath)
        trajs = p.values
    except Exception as e:
        trajs = []
        for i in range(100000 // 4096):
            states, actions, next_states, rewards = generate_traj(trainer_env, trainer_model, 4096)
            for se, ae, ne, re in zip(states, actions, next_states, rewards):
                trajs.append([])
                for s, a, n, r in zip(se, ae, ne, re):
                    trajs[-1].append(np.hstack([s, a, n, r]))
                trajs[-1] = np.vstack(trajs[-1])
        trajs = np.vstack(trajs)
        pandas.DataFrame(trajs).to_csv(trajs_fpath, index=False, header=False)

    del trainer_model

    model = ddpg(ddpg_policy, student_env, eval_env=eval_env, verbose=1)
    model.pretrain(trajs, trajs, n_epochs=100000 // 4096)
    model.learn(total_timesteps=nsteps)
