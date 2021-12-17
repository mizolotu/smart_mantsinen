import os
import tensorflow as tf
import numpy as np
import argparse as arp

from collections import deque
from config import ppo_net_arch, waypoints_dir, dataset_dir, signal_dir, lookback, tstep, batch_size, npretrain, patience, learning_rate, action_scale
from stable_baselines.ppo.policies import PPOPolicy
from common.data_utils import read_csv, load_waypoints_and_meta, load_signals
from gym.spaces import Box
from time import time

def dummy_predictor(x):
    actions = []
    for obs in x:
        a = (obs[-1, obs_features - act_dim - len(values_out) : obs_features - len(values_out)] * 2 - 1) * action_scale
        actions.append(a)
    actions = np.vstack(actions)
    return actions

if __name__ == '__main__':

    # parse args

    parser = arp.ArgumentParser()
    parser.add_argument('-g', '--gpu', help='GPU', default='-1')
    args = parser.parse_args()

    # gpu

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load data

    data_tr = read_csv(dataset_dir, 'train.csv')
    data_val = read_csv(dataset_dir, 'test.csv')
    _, _, data_tr_lens, data_val_lens = load_waypoints_and_meta(waypoints_dir, dataset_dir)

    # load signals

    values_in, xmin_in, xmax_in = load_signals(signal_dir, 'input')
    values_out, xmin_out, xmax_out = load_signals(signal_dir, 'output')

    # obs and act dim

    io_dim = len(values_in) + len(values_out)
    act_dim = len(values_in)
    obs_features = data_tr.shape[1] - act_dim - 1 - 3

    # spaces

    obs_space = Box(shape=(lookback, obs_features), low=-np.inf, high=np.inf)
    act_space = Box(shape=(act_dim,), low=-action_scale, high=action_scale)

    # create model

    model = PPOPolicy(
        obs_space,
        act_space,
        lambda x: learning_rate,
        vf_trainable=False,
        pi_trainable=False,
        net_arch=ppo_net_arch,
        activation_fn=tf.nn.tanh
    )
    model.summary()

    # data dims

    ntrain = len(data_tr_lens)
    nval = len(data_val_lens)
    print(f'Training on {ntrain} trajectories, validating on {nval}')
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

    nbatches_tr = nbatches_tr // batch_size
    nbatches_val = nbatches_val // batch_size

    print(f'Number of training batches: {nbatches_tr}, number of validation batches: {nbatches_val}')

    batch_generation_freq = 100
    val_losses = deque(maxlen=10)
    patience_count = 0
    val_loss_min = np.inf
    best_weights = None

    def generate_batch(r_list, io_list, a_list, t_list, w_list):
        n = len(t_list)
        X, Y, I = [], [], []
        while len(X) < batch_size:
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

    # training

    for epoch in range(npretrain):

        t_start = time()

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
                tape.watch(model.trainable_variables)
                actions, values, log_probs, action_logits = model.call(x, training=True)
                loss = tf.reduce_mean(tf.square(actions - y))
            train_loss += loss

            # Optimization step

            gradients = tape.gradient(loss, model.trainable_variables)

            # Clip grad norm

            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        val_loss = 0.0
        dummy_loss0 = 0.0
        dummy_loss1 = 0.0

        for x, y, I in batches_val:
            #x, y, I = generate_batch(r_val, io_val, a_val, t_val, w_val)
            actions, values, log_probs, action_logits = model.call(x)
            loss = tf.reduce_mean(tf.square(actions - y))
            val_loss += loss
            dummy_actions = dummy_predictor(x)
            loss_ = np.mean(tf.square(dummy_actions - y), axis=1)
            idx0 = np.where(I == 0)[0]
            idx1 = np.where(I == 1)[0]
            if len(idx0) > 0:
                dummy_loss0 += np.mean(loss_[idx0])
            if len(idx1) > 0:
                dummy_loss1 += np.mean(loss_[idx1])

        val_losses.append(val_loss / nbatches_val)

        print(f'At epoch {epoch + 1}/{npretrain}, train loss is {train_loss / nbatches_tr}, mean validation loss is {np.mean(val_losses)}, patience is {patience_count + 1}/{patience}, dummy losses are {dummy_loss0 / nbatches_val} and {dummy_loss1 / nbatches_val}, time elapsed: {time() - t_start}')

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
        f.write(f'{ppo_net_arch}, {lookback}, {tstep}: {val_loss_min}\n')