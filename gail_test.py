import pathlib, pickle, seals
import stable_baselines3 as sb3

from imitation.algorithms.adversarial import gail
from imitation.data import rollout
from imitation.rewards import reward_nets
from imitation.util import logger, util


def make_env(env_class):
  fn = lambda: env_class()
  return fn

if __name__ =='__main__':

    # Load pickled test demonstrations.

    with open("models/gym/cartpole_0/rollouts/final.pkl", "rb") as f:
        trajectories = pickle.load(f)

    print([item.rews.shape for item in trajectories])

    transitions = rollout.flatten_trajectories(trajectories)

    print(len(transitions))

    venv = util.make_vec_env("seals/CartPole-v0", n_envs=2)

    logdir = 'testlog'
    logdir_path = pathlib.Path(logdir)

    gail_logger = logger.configure(logdir_path / "GAIL/")
    gail_reward_net = reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )
    gail_trainer = gail.GAIL(
        venv=venv,
        demonstrations=transitions,
        demo_batch_size=32,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
        reward_net=gail_reward_net,
        custom_logger=gail_logger,
    )
    gail_trainer.train(total_timesteps=2048)
