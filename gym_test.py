import os, sys

from common import logger
from gym.envs.classic_control.pendulum import PendulumEnv
from common.gym_vec_env import SubprocVecEnv
from common.policies import MlpPolicy
from common.runners import Runner
from common.callbacks import CheckpointCallback
from baselines.ppo2.ppo2 import PPO2 as ppo
from common.model_utils import find_checkpoint_with_max_step
from common.base_class import ActorCriticRLModel

def make_env(env_class):
    fn = lambda: env_class()
    return fn

if __name__ == '__main__':

    nenvs = 4
    nsteps = 4000000

    logdir = 'models/gym/pendulum/ppo/'
    format_strs = os.getenv('', 'stdout,log,csv').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    env_fns = [make_env(PendulumEnv) for _ in range(nenvs)]
    env = SubprocVecEnv(env_fns)

    try:
        checkpoint_file = find_checkpoint_with_max_step('{0}/model_checkpoints/'.format(logdir))
        model = ppo.load(checkpoint_file)
        model.set_env(env)
        print('Model has been successfully loaded from {0}'.format(checkpoint_file))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        model = ppo(MlpPolicy, env, runner=Runner, verbose=1)
    finally:
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='{0}/model_checkpoints/'.format(logdir))
        model.learn(total_timesteps=nsteps, callback=[checkpoint_callback])
