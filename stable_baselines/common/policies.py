from itertools import zip_longest

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


class BasePolicy(Model):
    """
    The base policy object

    :param observation_space: (Gym Space) The observation space of the environment
    :param action_space: (Gym Space) The action space of the environment
    """

    def __init__(self, observation_space, action_space):
        super(BasePolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def save(self, path):
        """
        Save model to a given location.

        :param path: (str)
        """
        raise NotImplementedError()

    def load(self, path):
        """
        Load saved model from path.

        :param path: (str)
        """
        raise NotImplementedError()

    @tf.function
    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            current_var.assign((1. - tau) * current_var + tau * other_var)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)

    def call(self, x):
        raise NotImplementedError()


def create_mlp(input_dim, output_dim, net_arch, activation_fn=tf.nn.relu, squash_out=False):
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: (int) Dimension of the input vector
    :param output_dim: (int)
    :param net_arch: ([int]) Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: (tf.activations or str) The activation function
        to use after each layer.
    :param squash_out: (bool) Whether to squash the output using a Tanh
        activation function
    """
    modules = [layers.Flatten(input_shape=(input_dim,), dtype=tf.float32)]

    if len(net_arch) > 0:
        modules.append(layers.Dense(net_arch[0], activation=activation_fn))

    for idx in range(len(net_arch) - 1):
        modules.append(layers.Dense(net_arch[idx + 1], activation=activation_fn))

    if output_dim > 0:
        modules.append(layers.Dense(output_dim, activation=None))
    if squash_out:
        modules.append(layers.Activation(activation='tanh'))
    return modules


_policy_registry = dict()


def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]



def get_policy_from_name(base_policy_type, name):
    """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
    if base_policy_type not in _policy_registry:
        raise ValueError("Error: the policy type {} is not registered!".format(base_policy_type))
    if name not in _policy_registry[base_policy_type]:
        raise ValueError("Error: unknown policy type {}, the only registed policy type are: {}!"
                         .format(name, list(_policy_registry[base_policy_type].keys())))
    return _policy_registry[base_policy_type][name]


def register_policy(name, policy):
    """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError("Error: the policy {} is not of any known subclasses of BasePolicy!".format(policy))

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        raise ValueError("Error: the name {} is alreay registered for a different policy, will not override."
                         .format(name))
    _policy_registry[sub_class][name] = policy


class MlpExtractor(Model):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].


    :param feature_dim: (int) Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: (tf.nn.activation) The activation function to use for the networks.
    """
    def __init__(self, lookback, net_arch, activation_fn, batchnorm=True, shared_trainable=True, vf_trainable=True, pi_trainable=True, dropout=0.5):
        super(MlpExtractor, self).__init__()

        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network

        # Iterate through the shared layers and build the shared parts of the network

        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer

                # TODO: give layer a meaningful name
                # shared_net.append(layers.Dense(layer_size, input_shape=(last_layer_dim_shared,), activation=activation_fn))

                if batchnorm:
                    shared_net.append(layers.BatchNormalization(trainable=shared_trainable))
                shared_net.append(layers.Dense(layer_size, activation=activation_fn, trainable=shared_trainable))
                shared_net.append(layers.Dropout(dropout, trainable=shared_trainable))
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']

                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break  # From here on the network splits up in policy and value network

        shared_net.append(layers.Flatten())
        last_layer_dim_shared = layer_size * lookback

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network

        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(layers.Dense(pi_layer_size, input_shape=(last_layer_dim_pi,), activation=activation_fn, trainable=pi_trainable))
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(layers.Dense(vf_layer_size, input_shape=(last_layer_dim_vf,), activation=activation_fn, trainable=vf_trainable))
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module

        self.shared_net = Sequential(shared_net)
        self.policy_net = Sequential(policy_net)
        self.value_net = Sequential(value_net)

    def call(self, features):
        """
        :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


class FeatureExtractor(Model):

    def __init__(self, net_arch, activation_fn, shared_trainable=True, vf_trainable=True, pi_trainable=True, dropout=0.5, gn_std=0.01, l1=1e-4, l2=1e-5):
        super(FeatureExtractor, self).__init__()

        self.shared_trainable = shared_trainable
        self.vf_trainable = vf_trainable
        self.pi_trainable = pi_trainable

        self.shared_net, self.policy_net, self.value_net = [], [], []
        self.shared_net_bn, self.policy_net_bn, self.value_net_bn = [], [], []
        self.shared_net_gn = []
        self.shared_net_do = []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network

        # Iterate through the shared layers and build the shared parts of the network

        shared_part = [layer for layer in net_arch if isinstance(layer, tuple)]
        splitted_part = [layer for layer in net_arch if isinstance(layer, dict)]

        assert len(shared_part) > 0, 'shared part should contain at least one layer'
        assert shared_part[-1][0] == 'dense', 'The last layer in the shared part should be dense'

        for idx, layer in enumerate(shared_part[:-1]):
            if isinstance(layer, tuple):
                layer_type = layer[0]
                if layer_type == 'mask':
                    self.shared_net.append(layers.Masking())
                elif layer_type == 'lstm':
                    nunits = layer[1]
                    rseq = layer[2]
                    self.shared_net.append(layers.LSTM(
                        nunits, return_sequences=rseq, activation=activation_fn,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                        bias_regularizer=tf.keras.regularizers.l2(l2),
                        trainable=shared_trainable
                    ))
                elif layer_type == 'bilstm':
                    nunits = layer[1]
                    rseq = layer[2]
                    self.shared_net.append(layers.Bidirectional(layers.LSTM(
                        nunits, return_sequences=rseq, activation=activation_fn,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                        bias_regularizer=tf.keras.regularizers.l2(l2),
                        trainable=shared_trainable
                    )))
                elif layer_type == 'conv1d':
                    nfilters = layer[1]
                    kernel_size = layer[2]
                    stride_size = layer[3]
                    padding = layer[4]
                    self.shared_net.append(layers.Conv1D(
                        nfilters, kernel_size=kernel_size, strides=stride_size, padding=padding, activation=activation_fn,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                        bias_regularizer=tf.keras.regularizers.l2(l2),
                        trainable=shared_trainable
                    ))
                elif layer_type == 'dense':
                    nhidden = layer[1]
                    self.shared_net.append(layers.Dense(
                        nhidden, activation=activation_fn,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
                        bias_regularizer=tf.keras.regularizers.l2(l2),
                        trainable=shared_trainable
                    ))
                else:
                    raise NotImplemented

                # add batch normalization and dropout

                self.shared_net_bn.append(layers.BatchNormalization())
                self.shared_net_do.append(layers.Dropout(dropout))
                self.shared_net_gn.append(layers.GaussianNoise(gn_std))


        # last shared layer

        self.last_shared_layers = []
        self.last_shared_layers.append(layers.Flatten())
        nhidden = shared_part[-1][1]
        self.last_shared_layers.append(layers.Dense(
            nhidden, activation=activation_fn,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=tf.keras.regularizers.l2(l2),
            trainable=shared_trainable
        ))

        # add batch normalization and dropout

        self.last_shared_bn = layers.BatchNormalization()
        self.last_shared_gn = layers.GaussianNoise(gn_std)
        self.last_shared_do = layers.Dropout(dropout)

        # pi and vf streams

        last_layer_dim_pi = nhidden
        last_layer_dim_vf = nhidden

        for idx, layer in enumerate(splitted_part):

            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']

            break

        # Build the non-shared part of the network

        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                self.policy_net.append(layers.Dense(
                    pi_layer_size, input_shape=(last_layer_dim_pi,), activation=activation_fn, trainable=pi_trainable,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=tf.keras.regularizers.l2(l2),
                ))
                last_layer_dim_pi = pi_layer_size
                self.policy_net_bn.append(layers.BatchNormalization())

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                self.value_net.append(layers.Dense(
                    vf_layer_size, input_shape=(last_layer_dim_vf,), activation=activation_fn, trainable=vf_trainable,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2), bias_regularizer=tf.keras.regularizers.l2(l2),
                ))
                last_layer_dim_vf = vf_layer_size
                self.value_net_bn.append(layers.BatchNormalization())

        # save dim, used to create the distributions

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def call(self, features, training=False):

        shared_latent = features

        # shared layers

        for l, bn, gn, do in zip(self.shared_net, self.shared_net_bn, self.shared_net_gn, self.shared_net_do):
            shared_latent = bn(shared_latent, training & self.shared_trainable)
            shared_latent = gn(shared_latent, training & self.shared_trainable)
            shared_latent = l(shared_latent, training = training & self.shared_trainable)
            shared_latent = do(shared_latent, training & self.shared_trainable)

        # last shared

        shared_latent = self.last_shared_bn(shared_latent, training = training & self.shared_trainable)
        shared_latent = self.last_shared_gn(shared_latent, training & self.shared_trainable)
        for l in self.last_shared_layers:
            shared_latent = l(shared_latent)
        shared_latent = self.last_shared_do(shared_latent, training = training & self.shared_trainable)

        # pi layers

        pi_latent = shared_latent
        for l, bn in zip(self.policy_net, self.policy_net_bn):
            pi_latent = bn(pi_latent, training & self.pi_trainable)
            pi_latent = l(pi_latent)

        # vf layers

        vf_latent = shared_latent
        for l, bn in zip(self.value_net, self.value_net_bn):
            vf_latent = bn(vf_latent, training & self.vf_trainable)
            vf_latent = l(vf_latent)

        return pi_latent, vf_latent


class LstmExtractor(Model):

    def __init__(self, net_arch, activation_fn, shared_trainable=True, vf_trainable=True, pi_trainable=True,
                 dropout=0.5, gn_std=0.05):
        super(LstmExtractor, self).__init__()

        self.shared_trainable = shared_trainable
        self.vf_trainable = vf_trainable
        self.pi_trainable = pi_trainable

        self.shared_net, self.policy_net, self.value_net = [], [], []
        self.shared_net_bn, self.policy_net_bn, self.value_net_bn = [], [], []
        self.shared_net_do = []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network

        # Iterate through the shared layers and build the shared parts of the network

        shared_part = [layer for layer in net_arch if isinstance(layer, tuple)]
        splitted_part = [layer for layer in net_arch if isinstance(layer, dict)]

        assert len(shared_part) > 0, 'shared part should contain at least one layer'
        assert shared_part[-1][0] == 'dense', 'The last layer in the shared part should be dense'

        for idx, layer in enumerate(shared_part[:-1]):
            if isinstance(layer, tuple):
                layer_type = layer[0]
                if layer_type == 'lstm':
                    nunits = layer[1]
                    rseq = layer[2]
                    self.shared_net.append(layers.LSTM(
                        nunits, return_sequences=rseq, activation=activation_fn,
                        stateful=True, recurrent_dropout=dropout,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                        trainable=shared_trainable
                    ))
                elif layer_type == 'conv1d':
                    nfilters = layer[1]
                    kernel_size = layer[2]
                    stride_size = layer[3]
                    padding = layer[4]
                    self.shared_net.append(layers.Conv1D(
                        nfilters, kernel_size=kernel_size, strides=stride_size, padding=padding, activation=activation_fn,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                        trainable=shared_trainable
                    ))
                elif layer_type == 'dense':
                    nhidden = layer[1]
                    self.shared_net.append(layers.Dense(
                        nhidden, activation=activation_fn,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                        trainable=shared_trainable
                    ))
                else:
                    raise NotImplemented

                # add batch normalization and dropout

                self.shared_net_bn.append(layers.BatchNormalization())
                self.shared_net_do.append(layers.Dropout(dropout))


        # last shared layer

        self.last_shared_layers = []
        self.last_shared_layers.append(layers.Flatten())
        nhidden = shared_part[-1][1]
        self.last_shared_layers.append(layers.Dense(
            nhidden, activation=activation_fn,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=shared_trainable
        ))

        # add batch normalization and dropout

        self.last_shared_bn = layers.BatchNormalization()
        self.last_shared_do = layers.Dropout(dropout)

        # pi and vf streams

        last_layer_dim_pi = nhidden
        last_layer_dim_vf = nhidden

        for idx, layer in enumerate(splitted_part):

            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']

            break

        # Build the non-shared part of the network

        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                self.policy_net.append(layers.Dense(
                    pi_layer_size, input_shape=(last_layer_dim_pi,), activation=activation_fn, trainable=pi_trainable,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4),
                ))
                last_layer_dim_pi = pi_layer_size
                self.policy_net_bn.append(layers.BatchNormalization())

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                self.value_net.append(layers.Dense(
                    vf_layer_size, input_shape=(last_layer_dim_vf,), activation=activation_fn, trainable=vf_trainable,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=tf.keras.regularizers.l2(1e-4),
                ))
                last_layer_dim_vf = vf_layer_size
                self.value_net_bn.append(layers.BatchNormalization())

        # save dim, used to create the distributions

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # create networks

        #self.shared_net = Sequential(shared_net)
        #self.policy_net = Sequential(policy_net)
        #self.value_net = Sequential(value_net)

        #self.bn = layers.BatchNormalization()
        self.gn = layers.GaussianNoise(gn_std)
        #self.do = layers.Dropout(dropout)

    def reset_state(self):
        for layer in self.shared_net:
            if layer.name.startswith('lstm'):
                layer.reset_states()

    def call(self, features, training=False):

        # batch normazliation and gaussian noice

        #features = self.bn(features, training)
        features = self.gn(features, training & self.shared_trainable)

        # shared layers

        shared_latent = features
        for l, bn, do in zip(self.shared_net, self.shared_net_bn, self.shared_net_do):
            shared_latent = bn(shared_latent, training & self.shared_trainable)
            shared_latent = l(shared_latent, training = training & self.shared_trainable)
            shared_latent = do(shared_latent, training & self.shared_trainable)

        # last shared

        shared_latent = self.last_shared_bn(shared_latent, training = training & self.shared_trainable)
        for l in self.last_shared_layers:
            shared_latent = l(shared_latent)
        shared_latent = self.last_shared_do(shared_latent, training = training & self.shared_trainable)

        # pi layers

        pi_latent = shared_latent
        for l, bn in zip(self.policy_net, self.policy_net_bn):
            pi_latent = bn(pi_latent, training & self.pi_trainable)
            pi_latent = l(pi_latent)

        # vf layers

        vf_latent = shared_latent
        for l, bn in zip(self.value_net, self.value_net_bn):
            vf_latent = bn(vf_latent, training & self.vf_trainable)
            vf_latent = l(vf_latent)

        #return self.policy_net(shared_latent), self.value_net(shared_latent)
        return pi_latent, vf_latent