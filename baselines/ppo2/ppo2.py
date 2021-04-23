import time
import gym
import numpy as np
import tensorflow as tf

from common import logger
from common.math_util import explained_variance
from common.base_class import ActorCriticRLModel, SetVerbosity, TensorboardWriter
from common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from common.schedules import get_schedule_fn
from common.tf_util import total_episode_reward_logger, is_image, outer_scope_getter, make_session
from common.math_util import safe_mean

#L2_WEIGHT = .1
L2_WEIGHT = 0.0

class PPO2(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, runner, gamma=0.99, n_steps=2048, ent_coef=0.0, learning_rate=1e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=8, noptepochs=8, cliprange=0.1, cliprange_vf=None,
                 verbose=1, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):

        self.debug = False

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        self.n_batch = None
        self.summary = None

        self.runner_class = runner

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return self.runner_class(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action # policy.action will train logstd, but is it ok? not sure...

    def pretrain(self, data_tr, data_val, n_epochs=10, learning_rate=1e-4, adam_epsilon=1e-8, val_interval=None, l2_loss_weight=0.0, log_freq=100):

        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """

        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

        with self.graph.as_default():
            with tf.compat.v1.variable_scope('pretrain'):
                if continuous_actions:
                    obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                    policy_loss = tf.reduce_mean(input_tensor=tf.square(actions_ph - deterministic_actions_ph))
                    weight_params = [v for v in self.params if '/b' not in v.name]
                    l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])
                    loss = policy_loss + l2_loss_weight * l2_loss
                else:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    loss = tf.reduce_mean(input_tensor=loss)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.compat.v1.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with behavior cloning for {0} epochs on {1} samples of size {2}:".format(n_epochs, data_tr.shape[0], data_tr.shape[1]))

        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]

        ntrain = data_tr.shape[0]
        nbatches = ntrain // self.n_steps

        for epoch_idx in range(int(n_epochs)):

            if self.debug:
                old_vals = []
                with self.graph.as_default():
                    vars = tf.compat.v1.trainable_variables()
                    vars_vals = self.sess.run(vars)
                    for var, val in zip(vars, vars_vals):
                        old_vals.append(val)

            train_loss = 0.0

            # training

            for i in range(nbatches):
                idx = np.random.choice(ntrain, self.n_steps)
                expert_obs, expert_actions = data_tr[idx, :obs_dim], data_tr[idx, obs_dim:obs_dim+act_dim]
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions
                }
                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_
            train_loss /= nbatches

            # validation

            expert_obs, expert_actions = data_val[:, :obs_dim], data_val[:, obs_dim:obs_dim + act_dim]
            feed_dict = {
                obs_ph: expert_obs,
                actions_ph: expert_actions
            }

            val_loss, _ = self.sess.run([loss, optim_op], feed_dict)

            if self.verbose > 0 and (n_epochs <= log_freq or ((epoch_idx + 1) % (n_epochs // log_freq)) == 0):
                print('Epoch {0}/{1}: training loss = {2}, validation_loss = {3}'.format(epoch_idx + 1, n_epochs, train_loss, val_loss))

            if self.debug:
                with self.graph.as_default():
                    vars = tf.compat.v1.trainable_variables()
                    vars_vals = self.sess.run(vars)
                    for var, old_val, val in zip(vars, old_vals, vars_vals):
                        print('Var: {0}, difference: {1}'.format(var, np.linalg.norm(val - old_val)))

            del expert_obs, expert_actions

        if self.verbose > 0:
            print("Pretraining done!")

        return self

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common_.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps  # total number of samples in the batch

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.compat.v1.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")

                    self.another_action_ph = train_model.pdtype.sample_placeholder([None], name="another_action_ph")

                    self.advs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.compat.v1.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.compat.v1.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.compat.v1.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(input_tensor=train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.compat.v1.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                            tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(input_tensor=tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))

                    self.params = tf.compat.v1.trainable_variables()
                    weight_params = [v for v in self.params if '/b' not in v.name]
                    l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef + l2_loss * L2_WEIGHT

                    tf.compat.v1.summary.scalar('entropy_loss', self.entropy)
                    tf.compat.v1.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.compat.v1.summary.scalar('value_function_loss', self.vf_loss)
                    tf.compat.v1.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.compat.v1.summary.scalar('clip_factor', self.clipfrac)
                    tf.compat.v1.summary.scalar('loss', loss)

                    with tf.compat.v1.variable_scope('model'):
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.compat.v1.summary.histogram(var.name, var)
                    grads = tf.gradients(ys=loss, xs=self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.compat.v1.summary.scalar('discounted_rewards', tf.reduce_mean(input_tensor=self.rewards_ph))
                    tf.compat.v1.summary.scalar('learning_rate', tf.reduce_mean(input_tensor=self.learning_rate_ph))
                    tf.compat.v1.summary.scalar('advantage', tf.reduce_mean(input_tensor=self.advs_ph))
                    tf.compat.v1.summary.scalar('clip_range', tf.reduce_mean(input_tensor=self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.compat.v1.summary.scalar('clip_range_vf', tf.reduce_mean(input_tensor=self.clip_range_vf_ph))

                    tf.compat.v1.summary.scalar('old_neglog_action_probability', tf.reduce_mean(input_tensor=self.old_neglog_pac_ph))
                    tf.compat.v1.summary.scalar('old_value_pred', tf.reduce_mean(input_tensor=self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.compat.v1.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.compat.v1.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.compat.v1.summary.histogram('advantage', self.advs_ph)
                        tf.compat.v1.summary.histogram('clip_range', self.clip_range_ph)
                        tf.compat.v1.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.compat.v1.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if is_image(self.observation_space):
                            tf.compat.v1.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.compat.v1.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.compat.v1.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                with self.graph.as_default():
                    vars = tf.compat.v1.trainable_variables()
                    vars_vals = self.sess.run(vars)
                    for var, val in zip(vars, vars_vals):
                        print('Var: {0}'.format(var))

                #vars = tf.compat.v1.trainable_variables()
                #vars_vals = self.sess.run(vars)
                #for var, val in zip(vars, vars_vals):
                #    print("var: {}, value: {}".format(var.name, val))

                self.summary = tf.compat.v1.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm
2521
        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2", reset_num_timesteps=True):

        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):

                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )

                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # Unpack
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []

                if states is None:  # nonrecurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((epoch_num * self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_reward_c1_mean', safe_mean([ep_info['rc1'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_reward_c2_mean', safe_mean([ep_info['rc2'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_reward_c3_mean', safe_mean([ep_info['rc3'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

            callback.on_training_end()
            return self

    def demo(self, video_file=None):
        assert self.env.num_envs == 1, "You must pass only one environment when using this function"
        rollout = self.runner._run(video_file, headless=False)
        obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
        print('Reward: {0}'.format(np.mean(true_reward)))

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "runner": self.runner_class
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)