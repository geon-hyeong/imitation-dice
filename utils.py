from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network, utils
from tf_agents.specs.tensor_spec import TensorSpec
import random
import h5py
import os
from urllib import request
import gym

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2
SCALE_DIAG_MIN_MAX = (LOG_STD_MIN, LOG_STD_MAX)
MEAN_MIN_MAX = (-7, 7)
EPS = np.finfo(np.float32).eps
KEYS = ['observations', 'actions', 'rewards', 'terminals']


class TanhActor(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, name='TanhNormalPolicy',
                 mean_range=(-7., 7.), logstd_range=(-5., 2.), eps=EPS, initial_std_scaler=1,
                 kernel_initializer='he_normal', activation_fn=tf.nn.relu):
        self._input_specs = TensorSpec(state_dim)
        self._action_dim = action_dim
        self._initial_std_scaler = initial_std_scaler

        super(TanhActor, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=activation_fn,
                                           kernel_initializer=kernel_initializer, name='mlp')
        self._fc_mean = tf.keras.layers.Dense(action_dim, name='policy_mean/dense',
                                              kernel_initializer=kernel_initializer)
        self._fc_logstd = tf.keras.layers.Dense(action_dim, name='policy_logstd/dense',
                                                kernel_initializer=kernel_initializer)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, step_type=(), network_state=(), training=True):
        del step_type  # unused
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd) * self._initial_std_scaler
        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_action = pretanh_action_dist.sample()
        action = tf.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return (tf.tanh(mean), action, log_prob), network_state

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(tf.clip_by_value(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - action ** 2 + self.eps), axis=-1)

        return log_prob, pretanh_log_prob

    def get_log_prob(self, states, actions):
        """Evaluate log probs for actions conditined on states.
        Args:
            states: A batch of states.
            actions: A batch of actions to evaluate log probs on.
        Returns:
            Log probabilities of actions.
        """
        h = states
        for layer in self._fc_layers:
            h = layer(h, training=True)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd) * self._initial_std_scaler

        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_actions = tf.atanh(tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps))
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        log_probs = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs


class DiscreteActor(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, name='DiscretePolicy',
                 kernel_initializer='he_normal', activation_fn=tf.nn.relu):
        self._input_specs = TensorSpec(state_dim)
        self._action_dim = action_dim

        super(DiscreteActor, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=activation_fn, kernel_initializer=kernel_initializer, name='mlp')
        self._logit_layer = tf.keras.layers.Dense(action_dim, name='logits/dense', kernel_initializer=kernel_initializer)

    def call(self, inputs, step_type=(), network_state=(), training=True):
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)

        logits = self._logit_layer(h)
        dist = tfp.distributions.OneHotCategorical(logits)
        action = tf.cast(dist.sample(), tf.float32)
        greedy_action = tf.one_hot(tf.argmax(logits, axis=1), self._action_dim)
        log_prob = dist.log_prob(action)

        return (greedy_action, action, log_prob), network_state

    def get_log_prob(self, states, actions, training=True):
        """Evaluate log probs for actions conditined on states.
        Args:
          states: A batch of states.
          actions: A batch of actions to evaluate log probs on.
        Returns:
          Log probabilities of actions.
        """
        # h = tf.concat(states, axis=-1)
        h = states
        for layer in self._fc_layers:
            h = layer(h, training=training)

        logits = self._logit_layer(h)
        dist = tfp.distributions.OneHotCategorical(logits)

        log_probs = tf.expand_dims(dist.log_prob(actions), -1)  # To avoid broadcasting?

        return log_probs


class Critic(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, output_activation_fn=None, use_last_layer_bias=False,
                 output_dim=None, kernel_initializer='he_normal', name='ValueNetwork'):
        self._input_specs = TensorSpec(state_dim + action_dim)
        self._output_dim = output_dim

        super(Critic, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=tf.nn.relu,
                                           kernel_initializer=kernel_initializer, name='mlp')
        if use_last_layer_bias:
            last_layer_initializer = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
            self._last_layer = tf.keras.layers.Dense(output_dim or 1, activation=output_activation_fn,
                                                     kernel_initializer=last_layer_initializer,
                                                     bias_initializer=last_layer_initializer, name='value')
        else:
            self._last_layer = tf.keras.layers.Dense(output_dim or 1, activation=output_activation_fn, use_bias=False,
                                                     kernel_initializer=kernel_initializer, name='value')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        del step_type  # unused
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)
        h = self._last_layer(h)

        if self._output_dim is None:
            h = tf.reshape(h, [-1])

        return h, network_state


def load_d4rl_data(dirname, env_id, dataname, num_trajectories, start_idx=0, dtype=np.float32):
    MAX_EPISODE_STEPS = 1000

    original_env_id = env_id
    if env_id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2']:
        env_id = env_id.split('-v2')[0].lower()

    filename = f'{env_id}_{dataname}'
    filepath = os.path.join(dirname, filename + '.hdf5')
    # if not exists
    if not os.path.exists(filepath):
        os.makedirs(dirname, exist_ok=True)
        # Download the dataset
        remote_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}.hdf5'
        print(f'Download dataset from {remote_url} into {filepath} ...')
        request.urlretrieve(remote_url, filepath)
        print(f'Done!')

    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset_file = h5py.File(filepath, 'r')
    dataset_keys = KEYS
    use_timeouts = False
    use_next_obs = False
    if 'timeouts' in get_keys(dataset_file):
        if 'timeouts' not in dataset_keys:
            dataset_keys.append('timeouts')
        use_timeouts = True
    dataset = {k: dataset_file[k][:] for k in dataset_keys}
    dataset_file.close()
    N = dataset['observations'].shape[0]
    init_obs_, init_action_, obs_, action_, next_obs_, rew_, done_ = [], [], [], [], [], [], []
    episode_steps = 0
    num_episodes = 0
    for i in range(N - 1):
        if env_id == 'ant':
            obs = dataset['observations'][i][:27]
            if use_next_obs:
                next_obs = dataset['next_observations'][i][:27]
            else:
                next_obs = dataset['observations'][i + 1][:27]
        else:
            obs = dataset['observations'][i]
            if use_next_obs:
                next_obs = dataset['next_observations'][i]
            else:
                next_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            is_final_timestep = dataset['timeouts'][i]
        else:
            is_final_timestep = (episode_steps == MAX_EPISODE_STEPS - 1)

        if is_final_timestep:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
            continue

        if num_episodes >= start_idx:
            if episode_steps == 0:
                init_obs_.append(obs)
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            done_.append(done_bool)

        episode_steps += 1
        if done_bool:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break

    env = gym.make(original_env_id)
    if env.action_space.dtype == int:
        action_ = np.eye(env.action_space.n)[np.array(action_, dtype=np.int)]  # integer to one-hot encoding

    print(f'{num_episodes} trajectories are sampled')
    return np.array(init_obs_, dtype=dtype), np.array(obs_, dtype=dtype), np.array(action_, dtype=dtype), np.array(
        next_obs_, dtype=dtype), np.array(done_)


def add_absorbing_states(expert_states, expert_actions, expert_next_states,
                         expert_dones, env, dtype=np.float32):
    """Adds absorbing states to trajectories.
    Args:
      expert_states: A numpy array with expert states.
      expert_actions: A numpy array with expert states.
      expert_next_states: A numpy array with expert states.
      expert_dones: A numpy array with expert states.
      env: A gym environment.
    Returns:
        Numpy arrays that contain states, actions, next_states and dones.
    """

    # First add 0 indicator to all non-absorbing states.
    expert_states = np.pad(expert_states, ((0, 0), (0, 1)), mode='constant')
    expert_next_states = np.pad(
        expert_next_states, ((0, 0), (0, 1)), mode='constant')

    expert_states = [x for x in expert_states]
    expert_next_states = [x for x in expert_next_states]
    expert_actions = [x for x in expert_actions]
    expert_dones = [x for x in expert_dones]

    # Add absorbing states.
    i = 0
    current_len = 0
    while i < len(expert_states):
        current_len += 1
        if expert_dones[i] and current_len < env._max_episode_steps:  # pylint: disable=protected-access
            current_len = 0
            expert_states.insert(i + 1, env.get_absorbing_state())
            expert_next_states[i] = env.get_absorbing_state()
            expert_next_states.insert(i + 1, env.get_absorbing_state())
            action_dim = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0]
            expert_actions.insert(i + 1, np.zeros((action_dim,), dtype=dtype))
            expert_dones[i] = 0.0
            expert_dones.insert(i + 1, 1.0)
            i += 1
        i += 1

    expert_states = np.stack(expert_states)
    expert_next_states = np.stack(expert_next_states)
    expert_actions = np.stack(expert_actions)
    expert_dones = np.stack(expert_dones)

    return expert_states.astype(dtype), expert_actions.astype(dtype), expert_next_states.astype(dtype), expert_dones.astype(dtype)
