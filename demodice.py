import numpy as np
import tensorflow as tf
from tensorflow_gan.python.losses import losses_impl as tfgan_losses

import utils
import pickle
import os

EPS = np.finfo(np.float32).eps
EPS2 = 1e-3


class DemoDICE(tf.keras.layers.Layer):
    """ Class that implements DemoDICE training """
    def __init__(self, state_dim, action_dim, is_discrete_action: bool, config):
        super(DemoDICE, self).__init__()
        hidden_size = config['hidden_size']
        critic_lr = config['critic_lr']
        actor_lr = config['actor_lr']
        self.is_discrete_action = is_discrete_action
        self.grad_reg_coeffs = config['grad_reg_coeffs']
        self.discount = config['gamma']
        self.non_expert_regularization = config['alpha'] + 1.

        self.cost = utils.Critic(state_dim, action_dim, hidden_size=hidden_size,
                                 use_last_layer_bias=config['use_last_layer_bias_cost'],
                                 kernel_initializer=config['kernel_initializer'])
        self.critic = utils.Critic(state_dim, 0, hidden_size=hidden_size,
                                   use_last_layer_bias=config['use_last_layer_bias_critic'],
                                   kernel_initializer=config['kernel_initializer'])
        if self.is_discrete_action:
            self.actor = utils.DiscreteActor(state_dim, action_dim)
        else:
            self.actor = utils.TanhActor(state_dim, action_dim, hidden_size=hidden_size)

        self.cost.create_variables()
        self.critic.create_variables()
        self.actor.create_variables()

        self.cost_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    @tf.function
    def update(self, init_states, expert_states, expert_actions, expert_next_states,
               union_states, union_actions, union_next_states):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.cost.variables)
            tape.watch(self.actor.variables)
            tape.watch(self.critic.variables)

            # define inputs
            expert_inputs = tf.concat([expert_states, expert_actions], -1)
            union_inputs = tf.concat([union_states, union_actions], -1)

            # call cost functions
            expert_cost_val, _ = self.cost(expert_inputs)
            union_cost_val, _ = self.cost(union_inputs)
            unif_rand = tf.random.uniform(shape=(expert_states.shape[0], 1))
            mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
            mixed_inputs2 = unif_rand * tf.random.shuffle(union_inputs) + (1 - unif_rand) * union_inputs
            mixed_inputs = tf.concat([mixed_inputs1, mixed_inputs2], 0)

            # gradient penalty for cost
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(mixed_inputs)
                cost_output, _ = self.cost(mixed_inputs)
                cost_output = tf.math.log(1 / (tf.nn.sigmoid(cost_output) + EPS2) - 1 + EPS2)
            cost_mixed_grad = tape2.gradient(cost_output, [mixed_inputs])[0] + EPS
            cost_grad_penalty = tf.reduce_mean(
                tf.square(tf.norm(cost_mixed_grad, axis=-1, keepdims=True) - 1))
            cost_loss = tfgan_losses.minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.) \
                        + self.grad_reg_coeffs[0] * cost_grad_penalty
            union_cost = tf.math.log(1 / (tf.nn.sigmoid(union_cost_val) + EPS2) - 1 + EPS2)

            # nu learning
            init_nu, _ = self.critic(init_states)
            expert_nu, _ = self.critic(expert_states)
            expert_next_nu, _ = self.critic(expert_next_states)
            union_nu, _ = self.critic(union_states)
            union_next_nu, _ = self.critic(union_next_states)
            union_adv_nu = - tf.stop_gradient(union_cost) + self.discount * union_next_nu - union_nu

            non_linear_loss = self.non_expert_regularization * tf.reduce_logsumexp(
                union_adv_nu / self.non_expert_regularization)
            linear_loss = (1 - self.discount) * tf.reduce_mean(init_nu)
            nu_loss = non_linear_loss + linear_loss

            # weighted BC
            weight = tf.expand_dims(tf.math.exp(union_adv_nu / self.non_expert_regularization), 1)
            weight = weight / tf.reduce_mean(weight)
            pi_loss = - tf.reduce_mean(
                tf.stop_gradient(weight) * self.actor.get_log_prob(union_states, union_actions))

            # gradient penalty for nu
            if self.grad_reg_coeffs[1] is not None:
                unif_rand2 = tf.random.uniform(shape=(expert_states.shape[0], 1))
                nu_inter = unif_rand2 * expert_states + (1 - unif_rand2) * union_states
                nu_next_inter = unif_rand2 * expert_next_states + (1 - unif_rand2) * union_next_states

                nu_inter = tf.concat([union_states, nu_inter, nu_next_inter], 0)

                with tf.GradientTape(watch_accessed_variables=False) as tape3:
                    tape3.watch(nu_inter)
                    nu_output, _ = self.critic(nu_inter)

                nu_mixed_grad = tape3.gradient(nu_output, [nu_inter])[0] + EPS
                nu_grad_penalty = tf.reduce_mean(
                    tf.square(tf.norm(nu_mixed_grad, axis=-1, keepdims=True)))
                nu_loss += self.grad_reg_coeffs[1] * nu_grad_penalty

        nu_grads = tape.gradient(nu_loss, self.critic.variables)
        cost_grads = tape.gradient(cost_loss, self.cost.variables)
        pi_grads = tape.gradient(pi_loss, self.actor.variables)
        self.critic_optimizer.apply_gradients(zip(nu_grads, self.critic.variables))
        self.cost_optimizer.apply_gradients(zip(cost_grads, self.cost.variables))
        self.actor_optimizer.apply_gradients(zip(pi_grads, self.actor.variables))
        info_dict = {
            'cost_loss': cost_loss,
            'nu_loss': nu_loss,
            'actor_loss': pi_loss,
            'expert_nu': tf.reduce_mean(expert_nu),
            'union_nu': tf.reduce_mean(union_nu),
            'init_nu': tf.reduce_mean(init_nu),
            'union_adv': tf.reduce_mean(union_adv_nu),
        }
        del tape
        return info_dict

    @tf.function
    def step(self, observation, deterministic: bool = True):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        all_actions, _ = self.actor(observation)
        if deterministic:
            actions = all_actions[0]
        else:
            actions = all_actions[1]
        return actions


    def get_training_state(self):
        training_state = {
            'cost_params': [(variable.name, variable.value().numpy()) for variable in self.cost.variables],
            'critic_params': [(variable.name, variable.value().numpy()) for variable in self.critic.variables],
            'actor_params': [(variable.name, variable.value().numpy()) for variable in self.actor.variables],
            'cost_optimizer_state': [(variable.name, variable.value().numpy()) for variable in self.cost_optimizer.variables()],
            'critic_optimizer_state': [(variable.name, variable.value().numpy()) for variable in self.critic_optimizer.variables()],
            'actor_optimizer_state': [(variable.name, variable.value().numpy()) for variable in self.actor_optimizer.variables()],
        }
        return training_state

    def set_training_state(self, training_state):
        def _assign_values(variables, params):
            if len(variables) != len(params):
                import pdb; pdb.set_trace()
            assert len(variables) == len(params)
            for variable, (name, value) in zip(variables, params):
                assert variable.name == name
                variable.assign(value)

        _assign_values(self.cost.variables, training_state['cost_params'])
        _assign_values(self.critic.variables, training_state['critic_params'])
        _assign_values(self.actor.variables, training_state['actor_params'])
        _assign_values(self.cost_optimizer.variables(), training_state['cost_optimizer_state'])
        _assign_values(self.critic_optimizer.variables(), training_state['critic_optimizer_state'])
        _assign_values(self.actor_optimizer.variables(), training_state['actor_optimizer_state'])

    def save(self, filepath, training_info):
        print('Save checkpoint: ', filepath)
        training_state = self.get_training_state()
        data = {
            'training_state': training_state,
            'training_info': training_info,
        }
        with open(filepath + '.tmp', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(filepath + '.tmp', filepath)
        print('Saved!')

    def load(self, filepath):
        print('Load checkpoint:', filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.set_training_state(data['training_state'])
        return data
