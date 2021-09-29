import os, shutil
import os.path as osp

import gym, pyautogui, cv2
from gym import spaces
import tensorflow as tf
import numpy as np
import pandas as pd

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.buffers import RolloutBuffer
from stable_baselines.common.utils import explained_variance, get_schedule_fn
from stable_baselines.common import logger
from stable_baselines.ppo.policies import PPOPolicy
from stable_baselines.common.save_util import data_to_json, json_to_data

from common.solver_utils import get_solver_path, start_solver, stop_solver
from common.server_utils import is_backend_registered, delete_id
from time import sleep, time
from threading import Thread
from collections import deque

class PPOD(BaseRLModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI spinningup (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: (PPOPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: (float or callable) Clipping parameter, it can be a function of the current progress
        (from 1 to 0).
    :param clip_range_vf: (float or callable) Clipping parameter for the value function,
        it can be a function of the current progress (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param target_kl: (float) Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param seed: (int) Seed for the pseudo random generators
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """
    def __init__(self, policy, env, learning_rate=1e-4,
                 n_steps=2048, batch_size=64, n_epochs=16,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.1, clip_range_vf=None,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=None, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=0,
                 _init_setup_model=True, model_path=None, log_path=None, chkpt_name=None):

        super(PPOD, self).__init__(policy, env, PPOPolicy, policy_kwargs=policy_kwargs, verbose=verbose, create_eval_env=create_eval_env, support_multi_env=True, seed=seed)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef

        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.tb_writer = None

        self.iteration_start = 0
        self.time_elapsed_start = 0
        self.num_timesteps_start = 0
        self.reward_max = -np.inf

        self.model_path = model_path
        self.chkpt_name = chkpt_name
        params_loaded, policy_loaded = self._setup_model(model_path, chkpt_name)
        self.loaded = params_loaded & policy_loaded

        solverpath = get_solver_path()
        self._setup_runner(solverpath=solverpath)

        if log_path is not None:

            p = None
            if self.loaded:
                try:
                    print('Copying progress logs:\n')
                    fname = osp.join(log_path, 'progress.csv')
                    p = pd.read_csv(fname, delimiter=',', dtype=float)
                except Exception as e:
                    print(e)

            format_strs = os.getenv('', 'stdout,log,csv').split(',')
            logger.configure(os.path.abspath(log_path), format_strs)

            if p is not None:
                keys = p.keys()
                vals = p.values
                self.iteration_start = p['iterations'].values[-1]
                self.num_timesteps_start = p['total_timesteps'].values[-1]
                self.time_elapsed_start = p['time_elapsed'].values[-1]
                self.reward_max = np.nanmax(p['ep_reward_mean'].values)
                for i in range(vals.shape[0]):
                    for j in range(len(keys)):
                        logger.logkv(keys[j], vals[i, j])
                    logger.dumpkvs()

    def _setup_model(self, model_path=None, chkpt_name=None):

        self._setup_learning_rate()
        # TODO: preprocessing: one hot vector for obs discrete
        state_dim = np.prod(self.observation_space.shape)
        if isinstance(self.action_space, spaces.Box):
            # Action is a 1D vector
            action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
            # Action is a scalar
            action_dim = 1

        # TODO: different seed for each env when n_envs > 1
        if self.n_envs == 1:
            self.set_random_seed(self.seed)

        params_loaded = False
        if model_path is not None:
            if chkpt_name is not None:
                name = chkpt_name
            elif osp.isdir(osp.join(model_path, 'last')):
                name = 'last'
            elif osp.isdir(osp.join(model_path, 'first')):
                name = 'first'
            else:
                name = None

            try:
                data, w_path = self.load(model_path, name)
                self.__dict__.update(data)
                params_loaded = True
            except Exception as e:
                print(e)

        # rl policy

        self.policy = self.policy_class(self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs, shared_trainable=False)
        self.policy.summary()

        # sl policy

        self.pretrain_policy = self.policy_class(self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs, pi_trainable=False, vf_trainable=False)
        self.pretrain_policy.summary()

        policy_loaded = False
        if model_path is not None:
            try:
                self.policy.load(w_path)
                policy_loaded = True
                print(f'Model has been loaded from {w_path}')
            except Exception as e:
                print(e)

        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim, gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs)

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        return params_loaded, policy_loaded

    def _setup_runner(self, solverpath):

        self.solverpath = solverpath
        self.mvs = self.env.mvs
        self.dir = self.env.dir
        self.server = self.env.server
        self.recording = False
        self.is_solver_starting = False

        # copy model to different directories to deal with data.tmp bug

        if len(self.mvs) == 1:
            self.model_dirs = self.mvs
        else:
            self.model_dirs = []
            for i, mvs in enumerate(self.mvs):
                basename = []
                for j in range(3):
                    basename.append(osp.basename(mvs))
                    mvs = osp.dirname(mvs)
                env_i_dir = osp.join(self.dir[i], str(i))
                if not osp.isdir(env_i_dir):
                    shutil.copytree(mvs, env_i_dir)
                self.model_dirs.append(osp.abspath(osp.join(env_i_dir, *basename[::-1])))

    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        clipped_actions = self.policy.actor_forward(observation, deterministic=deterministic)
        #clipped_actions = self.policy.actor_forward(np.array(observation).reshape(1, -1), deterministic=deterministic)
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)
        return clipped_actions

    def collect_rollouts_(self, env, rollout_buffer, n_rollout_steps=256, callback=None, obs=None):

        n_steps = 0
        rollout_buffer.reset()
        rewards_ = []

        while n_steps < n_rollout_steps:
            actions, values, log_probs, _ = self.policy.call(obs)
            actions = actions.numpy()

            # Rescale and perform action

            clipped_actions = actions

            # Clip the actions to avoid out of bound error

            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            rewards_.append(rewards)
            self._update_info_buffer(infos)
            n_steps += 1
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(obs.reshape(self.n_envs, -1), actions, rewards, dones, values, log_probs)
            obs = new_obs

        rollout_buffer.compute_returns_and_advantage(values, dones=dones)
        self._update_reward_buffer(rewards_)

        return obs

    def _start(self, headless=False, sleep_interval=1):
        self.backend_procs = []
        self.start_times = []
        self.is_solver_starting = True
        for mvs, server in zip(self.model_dirs, self.server):
            proc = start_solver(self.solverpath, mvs, headless=headless)
            self.backend_procs.append(proc)
            while not is_backend_registered(server, proc.pid):
                sleep(sleep_interval)
            self.start_times.append(time())
        self.is_solver_starting = False

    def record(self, video_file, sleep_interval=0.04, x=210, y=90, width=755, height=400):
        screen_size = pyautogui.Size(width, height)
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(video_file, fourcc, 20.0, (screen_size))
        while self.recording:
            img = pyautogui.screenshot(region=(x, y, width, height))
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
            sleep(sleep_interval)
        cv2.destroyAllWindows()
        out.release()

    def _run_one(self, env_idx, mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, mb_rewards, last_values, deterministic=False, video_file=None,
                 sleep_interval=1, delay_interval=2, record_freq=20):

        # sleep to prevent pressure bug

        sleep(delay_interval)

        # reset env

        #print(f'Reseting {env_idx}')
        #print(f'In {env_idx}, time between register and reset: {time() - self.start_times[env_idx]}')
        obs = self.env.reset_one(env_idx)
        done = False
        #print(f'Solver {env_idx} has been reset')

        if video_file is not None:
            width = 755
            height = 400
            x = 210
            y = 90
            screen_size = pyautogui.Size(width, height)
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            out = cv2.VideoWriter(video_file, fourcc, 20.0, (screen_size))

        for step in range(self.n_steps):

            obs = obs.reshape(1, *obs.shape)
            actions, values, log_probs, _ = self.policy.call(obs, deterministic=deterministic)
            actions = actions.numpy()

            mb_obs[env_idx].append(obs[0])
            mb_actions[env_idx].append(actions[0])
            mb_values[env_idx].append(values[0])
            mb_neglogpacs[env_idx].append(log_probs[0])
            mb_dones[env_idx].append(done)

            # Rescale and perform action

            clipped_actions = actions

            # Clip the actions to avoid out of bound error

            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # perform the action

            obs, reward, done, info = self.env.step_one(env_idx, clipped_actions)

            mb_rewards[env_idx].append(reward)

            if video_file is not None and (step % record_freq) == 0:
                img = pyautogui.screenshot(region=(x, y, width, height))
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(frame)

            # reset if done

            if done:

                if video_file is not None:
                    cv2.destroyAllWindows()
                    out.release()

                #print(f'Env {env_idx} is done')
                stop_solver(self.backend_procs[env_idx])
                delete_id(self.server[env_idx], self.backend_procs[env_idx].pid)

                while self.is_solver_starting:
                    sleep(sleep_interval)
                self.is_solver_starting = True
                proc = start_solver(self.solverpath, self.model_dirs[env_idx], headless=False)
                self.backend_procs[env_idx] = proc
                while not is_backend_registered(self.server[env_idx], proc.pid):
                    sleep(sleep_interval)
                self.start_times[env_idx] = time()
                self.is_solver_starting = False
                sleep(delay_interval)
                self.env.set_attr('id', [proc.pid], indices=[env_idx])
                obs = self.env.reset_one(env_idx)

        obs = obs.reshape(1, *obs.shape)
        values = self.policy.value_forward(obs)
        last_values[env_idx] = values[0]

        stop_solver(self.backend_procs[env_idx])
        delete_id(self.server[env_idx], self.backend_procs[env_idx].pid)

    def collect_rollouts(self, rollout_buffer, deterministic=False, nenvs=None, video_file=None):

        if nenvs is None:
            nenvs = self.n_envs

        rollout_buffer.reset()
        self._start(headless=False)
        self.env.set_attr('id', [proc.pid for proc in self.backend_procs])

        mb_obs = [[] for _ in range(nenvs)]
        mb_actions = [[] for _ in range(nenvs)]
        mb_values = [[] for _ in range(nenvs)]
        mb_neglogpacs = [[] for _ in range(nenvs)]
        mb_dones = [[] for _ in range(nenvs)]
        mb_rewards = [[] for _ in range(nenvs)]
        last_values = [None for _ in range(nenvs)]

        threads = []
        for env_idx in range(nenvs):
            th = Thread(target=self._run_one, args=(env_idx, mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, mb_rewards, last_values, deterministic, video_file))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        mb_obs = [np.stack([mb_obs[idx][step] for idx in range(nenvs)]) for step in range(self.n_steps)]
        mb_rewards = [np.hstack([mb_rewards[idx][step] for idx in range(nenvs)]) for step in range(self.n_steps)]
        mb_actions = [np.vstack([mb_actions[idx][step] for idx in range(nenvs)]) for step in range(self.n_steps)]
        mb_values = [np.hstack([mb_values[idx][step] for idx in range(nenvs)]) for step in range(self.n_steps)]
        mb_neglogpacs = [np.hstack([mb_neglogpacs[idx][step] for idx in range(nenvs)]) for step in range(self.n_steps)]
        mb_dones = [np.hstack([mb_dones[idx][step] for idx in range(nenvs)]) for step in range(self.n_steps)]
        last_values = np.hstack(last_values)

        for obs, actions, rewards, dones, values, log_probs in zip(mb_obs, mb_actions, mb_rewards, mb_dones, mb_values, mb_neglogpacs):
            rollout_buffer.add(obs.reshape(nenvs, -1), actions, rewards, dones, values, log_probs)
        rollout_buffer.compute_returns_and_advantage(last_values, dones=mb_dones[-1])
        self._update_reward_buffer(mb_rewards)

        return obs

    @tf.function
    def policy_loss(self, advantage, log_prob, old_log_prob, clip_range):

        # Normalize advantage

        advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

        # ratio between old and new policy, should be one at the first iteration

        ratio = tf.exp(log_prob - old_log_prob)

        # clipped surrogate loss

        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range)

        return - tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

    @tf.function
    def value_loss(self, values, old_values, return_batch, clip_range_vf):
        if clip_range_vf is None:
            values_pred = values
        else:

            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling

            values_pred = old_values + tf.clip_by_value(values - old_values, -clip_range_vf, clip_range_vf)

        # Value loss using the TD(gae_lambda) target

        return tf.keras.losses.MSE(return_batch, values_pred)

    def train(self, gradient_steps, batch_size=64):

        # Compute current clip range

        clip_range = self.clip_range(self._current_progress)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress)
        else:
            clip_range_vf = None

        for gradient_step in range(gradient_steps):
            approx_kl_divs = []

            # Sample replay buffer

            for replay_data in self.rollout_buffer.get(batch_size):

                # Unpack

                obs, action, old_values, old_log_prob, advantage, return_batch = replay_data
                obs = obs.reshape(batch_size, *self.observation_space.shape)

                if isinstance(self.action_space, spaces.Discrete):

                    # Convert discrete action for float to long

                    action = action.astype(np.int64).flatten()

                with tf.GradientTape() as tape:
                    tape.watch(self.policy.trainable_variables)
                    values, log_prob, entropy = self.policy.evaluate_actions(obs, action)

                    # Flatten

                    values = tf.reshape(values, [-1])

                    policy_loss = self.policy_loss(advantage, log_prob, old_log_prob, clip_range)
                    value_loss = self.value_loss(values, old_values, return_batch, clip_range_vf)

                    # Entropy loss favor exploration

                    entropy_loss = - tf.reduce_mean(entropy)

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step

                gradients = tape.gradient(loss, self.policy.trainable_variables)

                # Clip grad norm

                gradients = [tf.clip_by_norm(gradient, self.max_grad_norm) for gradient in gradients]

                self.policy.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                approx_kl_divs.append(tf.reduce_mean(old_log_prob - log_prob).numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print("Early stopping at step {} due to reaching max kl: {:.2f}".format(gradient_step, np.mean(approx_kl_divs)))
                break

        explained_var = explained_variance(self.rollout_buffer.returns.flatten(),
                                           self.rollout_buffer.values.flatten())

        logger.logkv("clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.logkv("clip_range_vf", clip_range_vf)

        logger.logkv("explained_variance", explained_var)
        # TODO: gather stats for the entropy and other losses?
        logger.logkv("entropy", entropy.numpy().mean())
        logger.logkv("policy_loss", policy_loss.numpy())
        logger.logkv("value_loss", value_loss.numpy())

        if hasattr(self.policy, 'log_std'):
            logger.logkv("std", tf.exp(self.policy.log_std).numpy().mean())

    def pretrain(self, data_tr, data_val, nepochs=10000, patience=100):

        obs_dim = np.prod(self.observation_space.shape)
        act_dim = self.action_space.shape[0]

        ntrain = data_tr.shape[0]
        nval = data_val.shape[0]
        nbatches_tr = ntrain // self.batch_size
        nbatches_val = nval // self.batch_size

        val_losses = deque(maxlen=10)
        patience_count = 0
        val_loss_min = np.inf
        best_weights = None

        for epoch in range(nepochs):

            train_loss = 0.0

            for _ in range(nbatches_tr):

                idx = np.random.choice(ntrain, self.batch_size)
                expert_obs, expert_actions = data_tr[idx, :obs_dim], data_tr[idx, obs_dim:obs_dim + act_dim]

                expert_obs = expert_obs.reshape(self.batch_size, *self.observation_space.shape)

                if isinstance(self.action_space, spaces.Discrete):
                    actions_ = expert_actions[:, 0]
                elif isinstance(self.action_space, spaces.Box):
                    actions_ = expert_actions

                with tf.GradientTape() as tape:
                    tape.watch(self.pretrain_policy.trainable_variables)
                    actions, values, log_probs, action_logits = self.pretrain_policy.call(expert_obs)
                    if isinstance(self.action_space, spaces.Discrete):
                        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_, logits=action_logits))
                    elif isinstance(self.action_space, spaces.Box):
                        loss = tf.reduce_mean(tf.square(actions - actions_))
                train_loss += loss

                # Optimization step

                gradients = tape.gradient(loss, self.pretrain_policy.trainable_variables)

                # Clip grad norm

                self.pretrain_policy.optimizer.apply_gradients(zip(gradients, self.pretrain_policy.trainable_variables))

            val_loss = 0.0

            for _ in range(nbatches_val):

                idx = np.random.choice(nval, self.batch_size)
                expert_obs, expert_actions = data_val[idx, :obs_dim], data_val[idx, obs_dim:obs_dim + act_dim]

                expert_obs = expert_obs.reshape(self.batch_size, *self.observation_space.shape)

                if isinstance(self.action_space, spaces.Discrete):
                    actions_ = expert_actions[:, 0]
                elif isinstance(self.action_space, spaces.Box):
                    actions_ = expert_actions

                actions, values, log_probs, action_logits = self.pretrain_policy.call(expert_obs)
                if isinstance(self.action_space, spaces.Discrete):
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_, logits=action_logits))
                elif isinstance(self.action_space, spaces.Box):
                    loss = tf.reduce_mean(tf.square(actions - actions_))
                val_loss += loss

            val_losses.append(val_loss / nbatches_val)

            print(f'At epoch {epoch + 1}/{nepochs}, train loss is {train_loss / nbatches_tr}, validation loss is {val_loss / nbatches_val}, patience is {patience_count + 1}/{patience}')

            if np.mean(val_losses) < val_loss_min:
                val_loss_min = np.mean(val_losses)
                patience_count = 0
                best_weights = self.pretrain_policy.get_weights()
            else:
                patience_count += 1
                if patience_count >= patience:
                    self.policy.set_weights(best_weights)
                    break

    def save(self, path, name):

        # create dir

        w_dir = osp.join(path, name)
        if not osp.isdir(w_dir):
            os.mkdir(w_dir)

        # save policy weights
        w_path = osp.join(w_dir, 'model')
        self.policy.save(w_path)

        # save data

        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "gae_lambda": self.gae_lambda,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "verbose": self.verbose,
            "policy_class": self.policy_class,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "seed": self.seed,
            "policy_kwargs": self.policy_kwargs
        }
        d_path = osp.join(path, 'params')
        serialized_data = data_to_json(data)
        with open(d_path, 'w') as f:
            f.write(serialized_data)

    def load(self, path, name):

        # load params

        d_path = osp.join(path, 'params')
        with open(d_path, 'r') as f:
            json_data = f.read()
        data = json_to_data(json_data)

        #  weights

        w_path = osp.join(path, name, 'model')
        return data, w_path

    def learn(self, total_timesteps, callback=None, log_interval=1, eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="PPO", reset_num_timesteps=True):

        timesteps_since_eval, iteration, evaluations, obs, eval_env = self._setup_learn(eval_env, reset_env=False)
        iteration += self.iteration_start

        if self.tensorboard_log is not None:
            self.tb_writer = tf.summary.create_file_writer(os.path.join(self.tensorboard_log, f'{tb_log_name}_{time()}'))

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            obs = self.collect_rollouts(self.rollout_buffer)
            iteration += 1
            self.num_timesteps += self.n_steps * self.n_envs
            timesteps_since_eval += self.n_steps * self.n_envs
            self._update_current_progress(self.num_timesteps, total_timesteps)

            # Display training infos

            if self.verbose >= 1 and log_interval is not None and iteration % log_interval == 0:
                if len(self.ep_reward_buffer) > 0:
                    current_reward = self.safe_mean(self.ep_reward_buffer)
                    fps = int(self.num_timesteps / (time() - self.start_time))
                    logger.logkv("iterations", iteration)
                    logger.logkv('ep_reward_mean', current_reward)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time() - self.start_time + self.time_elapsed_start))
                    logger.logkv("total_timesteps", self.num_timesteps + self.num_timesteps_start)
                    logger.dumpkvs()
                    if iteration > self.iteration_start + 1:
                        self.save(self.model_path, 'last')
                    if len(self.ep_reward_buffer) == self.ep_reward_buffer.maxlen and current_reward > self.reward_max:
                        self.reward_max = current_reward
                        print(f'New best reward: {self.reward_max}')
                        self.save(self.model_path, 'best')

            self.train(self.n_epochs, batch_size=self.batch_size)

            # Evaluate the agent

            timesteps_since_eval = self._eval_policy(eval_freq, eval_env, n_eval_episodes, timesteps_since_eval, deterministic=True)

            # For tensorboard integration

            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    if len(self.ep_reward_buffer) > 0:
                        tf.summary.scalar('Reward', self.safe_mean(self.ep_reward_buffer), self.num_timesteps)

        return self

    def demo(self, ntests=1, video_file=None):
        self.ep_reward_buffer = deque(maxlen=ntests)
        for test in range(ntests):
            _ = self.collect_rollouts(self.rollout_buffer, deterministic=True, nenvs=1, video_file=video_file)
        print(f'Reward: {self.safe_mean(self.ep_reward_buffer)}')
