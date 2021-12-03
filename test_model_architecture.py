import tensorflow as tf
import numpy as np

from collections import deque
from config import ppo_net_arch, waypoints_dir, dataset_dir, signal_dir, lookback, tstep, batch_size, npretrain, patience, learning_rate, action_scale
from stable_baselines.ppo.policies import PPOPolicy
from common.data_utils import read_csv, load_waypoints_and_meta, load_signals
from gym.spaces import Box

if __name__ == '__main__':

    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load data

    data_tr = read_csv(dataset_dir, 'train.csv')
    data_val = read_csv(dataset_dir, 'test.csv')
    _, _, data_tr_lens, data_val_lens = load_waypoints_and_meta(waypoints_dir, dataset_dir)

    # load signals

    values, xmin, xmax = load_signals(signal_dir, 'input')

    # obs and act dim

    act_dim = len(values)
    obs_features = data_tr.shape[1] - act_dim - 1

    # spaces

    obs_space = Box(shape=(lookback, obs_features), low=-np.inf, high=np.inf)
    act_space = Box(shape=(act_dim,), low=-action_scale, high=action_scale)

    # create model

    model = PPOPolicy(obs_space, act_space, lambda x: learning_rate, vf_trainable=False, pi_trainable=False, net_arch=ppo_net_arch, activation_fn=tf.nn.tanh)
    model.summary()

    # data dims

    ntrain = len(data_tr_lens)
    nval = len(data_val_lens)
    print(f'Training on {ntrain} trajectories, validating on {nval}')

    # training batches

    x_tr, y_tr = [], []
    nbatches_tr = 0
    idx_start = 0
    batch_idx = 0
    for i, l in enumerate(data_tr_lens):
        idx = np.arange(idx_start, idx_start + l)
        expert_obs, expert_actions, expert_action_timestamps = data_tr[idx, :obs_features], data_tr[idx, obs_features:obs_features + act_dim], data_tr[idx, obs_features + act_dim]
        t = np.arange(expert_action_timestamps[0], expert_action_timestamps[-1], tstep)
        n = len(t)
        nbatches_tr += n
        if n > 0:
            x_tr.append(np.zeros((n, obs_features)))
            y_tr.append(np.zeros((n, act_dim)))
            for j in range(obs_features):
                x_tr[-1][:, j] = np.interp(t, expert_action_timestamps, expert_obs[:, j])
            for j in range(act_dim):
                y_tr[-1][:, j] = np.interp(t, expert_action_timestamps, expert_actions[:, j])
        batch_idx += 1
        idx_start = idx_start + l

    # validation batches

    x_val, y_val = [], []
    nbatches_val = 0
    idx_start = 0
    batch_idx = 0
    for i, l in enumerate(data_val_lens):
        idx = np.arange(idx_start, idx_start + l)
        expert_obs, expert_actions, expert_action_timestamps = data_val[idx, :obs_features], data_val[idx, obs_features:obs_features + act_dim], data_val[idx, obs_features + act_dim]
        t = np.arange(expert_action_timestamps[0], expert_action_timestamps[-1], tstep)
        n = len(t)
        nbatches_val += n
        x_val.append(np.zeros((n, obs_features)))
        y_val.append(np.zeros((n, act_dim)))
        for j in range(obs_features):
            x_val[-1][:, j] = np.interp(t, expert_action_timestamps, expert_obs[:, j])
        for j in range(act_dim):
            y_val[-1][:, j] = np.interp(t, expert_action_timestamps, expert_actions[:, j])
        batch_idx += 1
        idx_start = idx_start + l

    nbatches_tr = nbatches_tr // batch_size
    nbatches_val = nbatches_val // batch_size

    print(f'Number of training batches: {nbatches_tr}, number of validation batches: {nbatches_val}')

    val_losses = deque(maxlen=10)
    patience_count = 0
    val_loss_min = np.inf
    best_weights = None

    def generate_batch(x_list, y_list):
        n = len(x_list)
        x = np.zeros((batch_size, lookback, obs_features))
        y = np.zeros((batch_size, act_dim))
        for i in range(batch_size):
            traj_idx = np.random.choice(n)
            l = x_list[traj_idx].shape[0]
            idx_end = np.random.choice(l)
            idx_start = np.maximum(0, idx_end - lookback)
            x_ = x_list[traj_idx][idx_start:idx_end, :]
            x[i, :, :] = np.vstack([x_, np.zeros((lookback - x_.shape[0], obs_features))])
            y[i, :] = y_list[traj_idx][idx_end, :]
        return x, y

    for epoch in range(npretrain):

        train_loss = 0.0
        for i in range(nbatches_tr):
            x, y = generate_batch(x_tr, y_tr)
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                actions, values, log_probs, action_logits = model.call(x, training=True)
                loss = tf.reduce_mean(tf.square(actions - y))
            train_loss += loss

            # Optimization step

            gradients = tape.gradient(loss, model.trainable_variables)

            # Clip grad norm

            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        val_loss = 0.0

        for _ in range(nbatches_val):
            x, y = generate_batch(x_val, y_val)
            actions, values, log_probs, action_logits = model.call(x)
            loss = tf.reduce_mean(tf.square(actions - y))
            val_loss += loss

        val_losses.append(val_loss / nbatches_val)

        print(f'At epoch {epoch + 1}/{npretrain}, train loss is {train_loss / nbatches_tr}, mean validation loss is {np.mean(val_losses)}, patience is {patience_count + 1}/{patience}')

        if np.mean(val_losses) < val_loss_min:
            val_loss_min = np.mean(val_losses)
            patience_count = 0
            best_weights = model.get_weights()
        else:
            patience_count += 1
            if patience_count >= patience:
                model.set_weights(best_weights)
                print(f'Pretraining has finished with the minimum loss: {val_loss_min}')
                break
    with open('results.txt', 'a') as f:
        f.write(f'{ppo_net_arch}: {val_loss_min}\n')