import os, shutil, psutil
import os.path as osp

import gym, cv2
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

from common.server_utils import is_backend_registered, delete_id
from time import sleep, time
from threading import Thread
from collections import deque

try:
    import pyautogui
except:
    pass

from common.solver_utils import get_solver_path, start_solver, stop_solver

class PPOD(BaseRLModel):

    def __init__(self, policy, env, n_env_train,
                 learning_rate=2.5e-4, n_steps=2048, batch_size=64, n_epochs=32,
                 gamma=0.99, gae_lambda=0.95, clip_range=0.1, clip_range_vf=None,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 target_kl=None, tensorboard_log=None, create_eval_env=False,
                 policy_kwargs=None, verbose=0, seed=0,
                 _init_setup_model=True, model_path=None, log_path=None, chkpt_name=None):

        super(PPOD, self).__init__(policy, env, PPOPolicy, policy_kwargs=policy_kwargs, verbose=verbose, create_eval_env=create_eval_env, support_multi_env=True, seed=seed)

        self.n_envs_train = n_env_train

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
            action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.Discrete):
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

        #self.policy = self.policy_class(
        #    self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs, shared_trainable=False, batch_size=self.n_envs_train
        #)
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs, shared_trainable=False
        )
        self.policy.summary()

        policy_loaded = False
        if model_path is not None:
            try:
                self.policy.load(w_path)
                policy_loaded = True
                print(f'Model has been loaded from {w_path}')
            except Exception as e:
                print(e)

        self.rollout_buffer = RolloutBuffer(self.n_steps, state_dim, action_dim, gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs_train)

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

        clipped_actions = self.policy.actor_forward(observation, deterministic=deterministic)
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)
        return clipped_actions

    def _start(self, env_ids, headless=False, sleep_interval=1):
        self.backend_procs = [None for _ in range(len(env_ids))]
        self.start_times = [time() for _ in range(len(env_ids))]
        self.is_solver_starting = True
        for i, env_idx in enumerate(env_ids):
            mvs = self.model_dirs[env_idx]
            server = self.server[env_idx]
            proc = start_solver(self.solverpath, mvs, headless=headless)
            self.backend_procs[i] = proc
            while not is_backend_registered(server, proc.pid):
                sleep(sleep_interval)
            self.start_times[i] = time()
            self.env.set_attr('id', [proc.pid], indices=[env_idx])
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

    def _run_one(self, env_count, env_idx, mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones, mb_rewards, last_values, deterministic=False,
                 img_file=None, video_file=None, headless=False, sleep_interval=1, delay_interval=2, record_freq=4096, img_freq=4096):

        # sleep to prevent pressure bug

        sleep(delay_interval)

        # reset env
        obs = self.env.reset_one(env_idx)
        done = False

        if video_file is not None or img_file is not None:
            width = 755
            height = 400
            x = 210
            y = 90
            screen_size = pyautogui.Size(width, height)

            if img_file is not None:
                shots = []

            if video_file is not None:
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                out = cv2.VideoWriter(video_file, fourcc, 20.0, (screen_size))

        tstart = time()

        for step in range(self.n_steps):

            obs = obs.reshape(1, *obs.shape)
            actions, values, log_probs, _ = self.policy.call(obs, deterministic=deterministic)
            actions = actions.numpy()

            mb_obs[env_count].append(obs[0])
            mb_actions[env_count].append(actions[0])
            mb_values[env_count].append(values[0])
            mb_neglogpacs[env_count].append(log_probs[0])
            mb_dones[env_count].append(done)

            # Rescale and perform action

            clipped_actions = actions

            # Clip the actions to avoid out of bound error

            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # perform the action

            obs, reward, done, info = self.env.step_one(env_idx, clipped_actions)

            mb_rewards[env_count].append(reward)

            if video_file is not None and (step % record_freq) == 0:
                img = pyautogui.screenshot(region=(x, y, width, height))
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(frame)

            if img_file is not None and (step == 0 or (step + 1) % img_freq == 0):
                img = pyautogui.screenshot(region=(x, y, width, height))
                shots.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

            #print(step, time() - tstart)

            # reset if done

            if done:

                #print(f'Env {env_idx} is done')
                stop_solver(self.backend_procs[env_idx])
                delete_id(self.server[env_idx], self.backend_procs[env_idx].pid)

                while self.is_solver_starting:
                    sleep(sleep_interval)
                self.is_solver_starting = True
                proc = start_solver(self.solverpath, self.model_dirs[env_idx], headless)
                self.backend_procs[env_idx] = proc
                while not is_backend_registered(self.server[env_idx], proc.pid):
                    sleep(sleep_interval)
                self.start_times[env_idx] = time()
                self.is_solver_starting = False
                sleep(delay_interval)
                self.env.set_attr('id', [proc.pid], indices=[env_idx])
                obs = self.env.reset_one(env_idx)

            tstart = time()

        obs = obs.reshape(1, *obs.shape)
        values = self.policy.value_forward(obs)
        last_values[env_count] = values[0]

        if video_file is not None:
            cv2.destroyAllWindows()
            out.release()

        if img_file is not None:
            shots = cv2.vconcat(shots)
            cv2.imwrite(img_file, shots)

        #stop_solver(self.backend_procs[env_idx])
        #delete_id(self.server[env_idx], self.backend_procs[env_idx].pid)

    def collect_rollouts(self, rollout_buffer, headless=False, deterministic=False, env_ids=None, img_file=None, video_file=None, update_reward=False):

        if env_ids is None:
            env_ids = np.arange(self.n_envs_train)
        nenvs = len(env_ids)

        rollout_buffer.reset()
        self._start(env_ids, headless)

        mb_obs = [[] for _ in range(nenvs)]
        mb_actions = [[] for _ in range(nenvs)]
        mb_values = [[] for _ in range(nenvs)]
        mb_neglogpacs = [[] for _ in range(nenvs)]
        mb_dones = [[] for _ in range(nenvs)]
        mb_rewards = [[] for _ in range(nenvs)]
        last_values = [None for _ in range(nenvs)]

        threads = []
        for env_count, env_idx in enumerate(env_ids):
            th = Thread(target=self._run_one, args=(env_count, env_idx, mb_obs, mb_actions, mb_values, mb_neglogpacs, mb_dones,
                                                    mb_rewards, last_values, deterministic, img_file, video_file, headless))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        for env_count, env_idx in enumerate(env_ids):
            stop_solver(self.backend_procs[env_count])
            delete_id(self.server[env_idx], self.backend_procs[env_count].pid)

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
        if update_reward:
            self._update_reward_buffer(mb_rewards)
        else:
            print(f'Reward: {np.mean(mb_rewards)}')

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
                    values, log_prob, entropy = self.policy.evaluate_actions(obs, action, training=True)

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

    def pretrain(self, data_tr, data_val, data_tr_lens, data_val_lens, tstep, nepochs=10000, patience=100, batch_generation_freq=100):

        self.pretrain_policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate,  **self.policy_kwargs, pi_trainable=False, vf_trainable=False
        )
        self.pretrain_policy.summary()

        lookback = self.observation_space.shape[0]
        obs_features = self.observation_space.shape[1]

        act_dim = self.action_space.shape[0]

        assert data_tr.shape[1] == obs_features + act_dim + 1 + 3, 'Incorrect training data shape'
        assert data_val.shape[1] == obs_features + act_dim + 1 + 3, 'Incorrect validation data shape'

        ntrain = len(data_tr_lens)
        nval = len(data_val_lens)
        print(f'Training on {ntrain} trajectories, validating on {nval}')
        io_dim = obs_features - 3
        print(io_dim)
        spl_idx = [3, 3 + io_dim, 3 + io_dim + act_dim, 3 + io_dim + act_dim + 1]

        # training batches

        r_tr, io_tr, a_tr, t_tr, w_tr = [], [], [], [], []
        nbatches_tr = 0
        idx_start = 0
        batch_idx = 0
        for i, l in enumerate(data_tr_lens):
            idx = np.arange(idx_start, idx_start + l)
            expert_r, expert_io, expert_a, expert_t, expert_w = np.split(data_tr[idx, :], spl_idx, axis=1)
            expert_t = expert_t.flatten()
            n = len(expert_t)
            nbatches_tr += n
            if n > 0:
                r_tr.append(expert_r)
                io_tr.append(expert_io)
                a_tr.append(expert_a)
                t_tr.append(expert_t)
                w_tr.append(expert_w)
            batch_idx += 1
            idx_start = idx_start + l

        # validation batches

        r_val, io_val, a_val, t_val, w_val = [], [], [], [], []
        nbatches_val = 0
        idx_start = 0
        batch_idx = 0
        for i, l in enumerate(data_val_lens):
            idx = np.arange(idx_start, idx_start + l)
            expert_r, expert_io, expert_a, expert_t, expert_w = np.split(data_val[idx, :], spl_idx, axis=1)
            expert_t = expert_t.flatten()
            n = len(expert_t)
            nbatches_val += n
            if n > 0:
                r_val.append(expert_r)
                io_val.append(expert_io)
                a_val.append(expert_a)
                t_val.append(expert_t)
                w_val.append(expert_w)
            batch_idx += 1
            idx_start = idx_start + l

        nbatches_tr = nbatches_tr // self.batch_size
        nbatches_val = nbatches_val // self.batch_size

        print(f'Number of training batches: {nbatches_tr}, number of validation batches: {nbatches_val}')

        val_losses = deque(maxlen=10)
        patience_count = 0
        val_loss_min = np.inf
        best_weights = None

        def generate_batch(r_list, io_list, a_list, t_list, w_list):
            n = len(t_list)
            X, Y, I = [], [], []
            while len(X) < self.batch_size:
                traj_idx = np.random.choice(n)
                l = r_list[traj_idx].shape[0]
                idx_action = np.random.choice(l)
                t_action = t_list[traj_idx][idx_action]
                w_action = w_list[traj_idx][idx_action, :]
                t_start = t_action - lookback * tstep
                t = np.arange(t_start, t_action, tstep)[:lookback]
                t = t[np.where(t >= t_list[traj_idx][0])]
                t_idx = np.where(t_list[traj_idx] < t_start)[0]
                if len(t_idx) > 0:
                    idx_start = t_idx[-1]
                else:
                    idx_start = 0
                if idx_start < idx_action and len(t) > 0:

                    # w - xyz

                    r_ = np.zeros((len(t), 3))
                    for j in range(3):
                        r_[:, j] = np.interp(t, t_list[traj_idx][idx_start:idx_action], r_list[traj_idx][idx_start:idx_action, j])
                    r = np.vstack([r_list[traj_idx][0, :] * np.ones(lookback - r_.shape[0])[:, None], r_])
                    r = w_action - r

                    # io

                    io_ = np.zeros((len(t), io_dim))
                    for j in range(io_dim):
                        io_[:, j] = np.interp(t, t_list[traj_idx][idx_start:idx_action], io_list[traj_idx][idx_start:idx_action, j])
                    io = np.vstack([io_list[traj_idx][0, :] * np.ones(lookback - io_.shape[0])[:, None], io_])

                    # x and y

                    x = np.hstack([r, io])
                    y = a_list[traj_idx][idx_action, :]
                    X.append(x)
                    Y.append(y)
                    if len(t) < lookback:
                        I.append(0)
                    else:
                        I.append(1)
            X = np.array(X)
            Y = np.vstack(Y)
            I = np.array(I)
            return X, Y, I

        for epoch in range(nepochs):

            if epoch % batch_generation_freq == 0:
                batches_tr, batches_val = [], []
                for i in range(nbatches_tr):
                    x, y, I = generate_batch(r_tr, io_tr, a_tr, t_tr, w_tr)
                    batches_tr.append((x, y, I))
                for i in range(nbatches_val):
                    x, y, I = generate_batch(r_val, io_val, a_val, t_val, w_val)
                    batches_val.append((x, y, I))

            train_loss = 0.0
            for x, y, _ in batches_tr:
                #x, y, _ = generate_batch(r_tr, io_tr, a_tr, t_tr, w_tr)
                with tf.GradientTape() as tape:
                    tape.watch(self.pretrain_policy.trainable_variables)
                    actions, values, log_probs, action_logits = self.pretrain_policy.call(x, training=True)
                    if isinstance(self.action_space, spaces.Discrete):
                        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=action_logits))
                    elif isinstance(self.action_space, spaces.Box):
                        loss = tf.reduce_mean(tf.square(actions - y))
                train_loss += loss

                # Optimization step

                gradients = tape.gradient(loss, self.pretrain_policy.trainable_variables)

                # Clip grad norm

                self.pretrain_policy.optimizer.apply_gradients(zip(gradients, self.pretrain_policy.trainable_variables))

            val_loss = 0.0

            for x, y, I in batches_val:
                #x, y, _ = generate_batch(r_val, io_val, a_val, t_val, w_val)
                actions, values, log_probs, action_logits = self.pretrain_policy.call(x)
                if isinstance(self.action_space, spaces.Discrete):
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=action_logits))
                elif isinstance(self.action_space, spaces.Box):
                    loss = tf.reduce_mean(tf.square(actions - y))
                val_loss += loss

            val_losses.append(val_loss / nbatches_val)

            print(f'At epoch {epoch + 1}/{nepochs}, train loss is {train_loss / nbatches_tr}, mean validation loss is {np.mean(val_losses)}, patience is {patience_count + 1}/{patience}')

            if np.mean(val_losses) < val_loss_min:
                val_loss_min = np.mean(val_losses)
                patience_count = 0
                best_weights = self.pretrain_policy.get_weights()
            else:
                patience_count += 1
                if patience_count >= patience:
                    self.policy.set_weights(best_weights)
                    print(f'Pretraining has finished with the minimum loss: {val_loss_min}')
                    break

    def pretrain_recurrent(self, data_tr, data_val, data_tr_lens, data_val_lens, nepochs=10000, patience=100):

        obs_dim = np.prod(self.observation_space.shape)
        act_dim = self.action_space.shape[0]

        ntrain = len(data_tr_lens)
        nval = len(data_val_lens)
        print(f'Training on {ntrain} samples, validating on {nval} samples:')

        # batches

        tr_traj_len_max = np.max(data_tr_lens)
        nbatches_tr = np.ceil(tr_traj_len_max / self.batch_size)
        val_traj_len_max = np.max(data_val_lens)
        nbatches_val = np.ceil(val_traj_len_max / self.batch_size)

        # model

        self.tr_policy = self.policy_class(self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs, pi_trainable=False, vf_trainable=False, batch_size=ntrain, nsteps=self.batch_size)
        self.val_policy = self.policy_class(self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs, pi_trainable=False, vf_trainable=False, batch_size=nval, nsteps=self.batch_size)

        # training batches

        x_tr = np.zeros((ntrain, nbatches_tr * self.batch_size, obs_dim))
        y_tr = np.zeros((ntrain, nbatches_val * self.batch_size, act_dim))
        idx_start = 0
        batch_idx = 0
        for l in data_tr_lens:
            idx = np.arange(idx_start, idx_start + l)
            expert_obs, expert_actions = data_tr[idx, :obs_dim], data_tr[idx, obs_dim:obs_dim + act_dim]
            x_tr[batch_idx, :len(idx), :] = expert_obs
            y_tr[batch_idx, :len(idx), :] = expert_actions
            batch_idx += 1
            idx_start = idx_start + l

        # validation batches

        x_val = np.zeros((nval, val_traj_len_max, obs_dim))
        y_val = np.zeros((nval, val_traj_len_max, act_dim))
        idx_start = 0
        batch_idx = 0
        for l in data_val_lens:
            idx = np.arange(idx_start, idx_start + l)
            expert_obs, expert_actions = data_val[idx, :obs_dim], data_val[idx, obs_dim:obs_dim + act_dim]
            x_val[batch_idx, :len(idx), :] = expert_obs
            y_val[batch_idx, :len(idx), :] = expert_actions
            batch_idx += 1
            idx_start = idx_start + l

        # early stoping init

        val_losses = deque(maxlen=10)
        patience_count = 0
        val_loss_min = np.inf
        best_weights = None

        # main loop

        for epoch in range(nepochs):

            # training

            train_loss = 0.0
            for i in range(nbatches_tr):
                x, y = x_tr[:, i*self.batch_size:(i+1)*self.batch_size, :], y_tr[:, i, :]
                with tf.GradientTape() as tape:
                    tape.watch(self.tr_policy.trainable_variables)
                    actions, values, log_probs, action_logits = self.tr_policy.call(x, training=True)
                    loss = tf.reduce_mean(tf.square(actions - y))
                train_loss += loss

                # Optimization step

                gradients = tape.gradient(loss, self.tr_policy.trainable_variables)

                # Clip grad norm

                self.tr_policy.optimizer.apply_gradients(zip(gradients, self.tr_policy.trainable_variables))

            self.tr_policy.features_extractor.reset_state()

            # transfer weights

            self.val_policy.set_weights(self.tr_policy.get_weights())

            # validation

            val_loss = 0.0
            for i in range(val_traj_len_max):
                x, y = x_val[:, i:i + 1, :], y_val[:, i, :]
                actions, values, log_probs, action_logits = self.val_policy.call(x)
                loss = tf.reduce_mean(tf.square(actions - y))
                val_loss += loss

            self.val_policy.features_extractor.reset_state()

            val_losses.append(val_loss / (nval * val_traj_len_max))

            print(f'At epoch {epoch + 1}/{nepochs}, train loss is {train_loss / (ntrain * tr_traj_len_max)}, '
                  f'validation loss is {val_loss / (nval * val_traj_len_max)}, patience is {patience_count + 1}/{patience}')

            if np.mean(val_losses) < val_loss_min:
                val_loss_min = np.mean(val_losses)
                patience_count = 0
                best_weights = self.tr_policy.get_weights()
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
            "n_envs_train": self.n_envs_train,
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

    def learn(self, total_timesteps, callback=None, log_interval=1, eval_env=None, eval_freq=1, tb_log_name="PPO", reset_num_timesteps=True):

        timesteps_since_eval, iteration, evaluations, obs, eval_env = self._setup_learn(eval_env, reset_env=False)
        iteration += self.iteration_start

        if self.tensorboard_log is not None:
            self.tb_writer = tf.summary.create_file_writer(os.path.join(self.tensorboard_log, f'{tb_log_name}_{time()}'))

        while self.num_timesteps < total_timesteps:

            self.collect_rollouts(self.rollout_buffer, env_ids=np.arange(self.n_envs_train), headless=True, update_reward=False)
            iteration += 1

            self.num_timesteps += self.n_steps * self.n_envs_train
            timesteps_since_eval += self.n_steps * self.n_envs_train
            self._update_current_progress(self.num_timesteps, total_timesteps)
            self.train(self.n_epochs, batch_size=self.batch_size)

            # Evaluate the agent

            if iteration == 1 or iteration % eval_freq == 0:
                _ = self.collect_rollouts(self.rollout_buffer, deterministic=True, env_ids=np.arange(self.n_envs_train, self.n_envs), update_reward=True)

            # Display learning info

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

            # For tensorboard integration

            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    if len(self.ep_reward_buffer) > 0:
                        tf.summary.scalar('Reward', self.safe_mean(self.ep_reward_buffer), self.num_timesteps)

        return self

    def demo(self, ntests=1, img_file=None, video_file=None):
        self.ep_reward_buffer = deque(maxlen=ntests)
        for test in range(ntests):
            _ = self.collect_rollouts(self.rollout_buffer, deterministic=True, env_ids=[0], img_file=img_file, video_file=video_file, update_reward=True)
        print(f'Reward: {self.safe_mean(self.ep_reward_buffer)}')
