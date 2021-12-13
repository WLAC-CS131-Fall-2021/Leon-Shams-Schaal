import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow_addons.layers import NoisyDense

np.random.seed(0)
tf.random.set_seed(0)


class ReplayMemory:
    size = 0

    def __init__(self, max_size, state_shape, n_step):
        self.max_size = max_size
        self.n_step = n_step

        self.states = np.zeros((self.max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(self.max_size, dtype=np.int32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.terminals = np.zeros(self.max_size, dtype=np.bool)

    def store_transition(self, state, action, reward, terminal):
        idx = self.size % self.max_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminals[idx] = terminal

        self.size += 1

    def sample_minibatch(self, minibatch_size, discount):
        minibatch_idxs = np.random.randint(
            0, min(self.size, self.max_size) - self.n_step, minibatch_size
        )

        terminal_slices = self.terminals[
            np.repeat(minibatch_idxs, self.n_step)
            + np.tile(np.arange(self.n_step), len(minibatch_idxs))
        ].reshape(-1, self.n_step)
        terminal_idxs = terminal_slices.sum(-1) > 0
        num_steps = np.full(len(minibatch_idxs), self.n_step)
        num_steps[terminal_idxs] = np.flatnonzero(terminal_slices[terminal_idxs]) + 1

        current_states = self.states[minibatch_idxs]
        actions = self.actions[minibatch_idxs + 1]
        rewards = np.array(
            [
                sum(self.rewards[start_idx : start_idx + n] * discount ** np.arange(n))
                for n, start_idx in zip(num_steps, minibatch_idxs + 1)
            ],
            dtype=self.rewards.dtype,
        )
        terminals = self.terminals[minibatch_idxs + num_steps]
        next_states = self.states[minibatch_idxs + num_steps]

        return (
            current_states,
            actions,
            rewards,
            terminals,
            next_states,
            num_steps.astype(np.float32),
        )


class DuelingNetwork(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu", kernel_initializer="he_uniform")
        self.dense2 = Dense(256, activation="relu", kernel_initializer="he_uniform")
        self.dense3 = Dense(256, activation="relu", kernel_initializer="he_uniform")

        self.V = NoisyDense(1)
        self.A = NoisyDense(action_space)

    @tf.function
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        value = self.V(x)
        advantage = self.A(x)

        Q = value + tf.subtract(
            advantage, tf.reduce_mean(advantage, axis=1, keepdims=True)
        )
        return Q

    @tf.function
    def reset_noise(self):
        self.V.reset_noise()
        self.A.reset_noise()


class Agent:
    discount = 0.99

    def __init__(
        self,
        replay_memory,
        minibatch_size,
        action_space,
        min_rp_mem_size,
        target_model_update_frequency,
        max_no_op_steps,
    ):
        self.rp_mem = replay_memory
        self.minibatch_size = minibatch_size

        self.action_space = action_space
        self.online_model = DuelingNetwork(self.action_space)
        self.target_model = DuelingNetwork(self.action_space)

        state_shape = self.rp_mem.states.shape[1:]
        self.online_model.build((self.minibatch_size, *state_shape))
        self.target_model.build((self.minibatch_size, *state_shape))
        self.update_target_model()

        self.loss_fn = tf.keras.losses.get("mse")
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, epsilon=1.5e-04, clipnorm=10
        )

        self.min_rp_mem_size = max(min_rp_mem_size, self.minibatch_size)
        self.target_model_update_frequency = target_model_update_frequency

        self.max_no_op_steps = max_no_op_steps
        self.no_op_steps = np.random.randint(self.max_no_op_steps)

    def select_action(self, state):
        # Perform random action if there was a terminal in the last `self.no_op_steps`
        rp_mem_size = self.rp_mem.size % self.rp_mem.max_size
        last_no_op_idxs = np.arange(rp_mem_size - self.no_op_steps, rp_mem_size)
        if any(self.rp_mem.terminals[last_no_op_idxs]):
            return np.random.randint(self.action_space)

        # Take the action greedily
        predicted_rewards = self.online_model(state[np.newaxis, ...])
        return np.argmax(predicted_rewards)

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def train(self, *transition):
        self.rp_mem.store_transition(*transition)

        # If terminal, randomly choose the number of no-op steps for the next episode
        if transition[3]:
            self.no_op_steps = np.random.randint(self.max_no_op_steps)

        # Train if our replay memory is large enough for a minibatch
        if self.rp_mem.size > self.min_rp_mem_size:
            # Update the target model after a certain number of steps
            if not self.rp_mem.size % self.target_model_update_frequency:
                self.update_target_model()

            # Train the online model
            minibatch = self.rp_mem.sample_minibatch(self.minibatch_size, self.discount)
            self.train_step(*minibatch)

        self.online_model.reset_noise()
        self.target_model.reset_noise()

    @tf.function
    def train_step(
        self, current_states, actions, rewards, terminals, next_states, num_steps
    ):
        next_qs = self.online_model(next_states)
        next_actions = tf.argmax(next_qs, axis=1)

        target_next_qs = self.target_model(next_states)
        masked_tnqs = target_next_qs * tf.one_hot(next_actions, self.action_space)
        next_actions_tnqs = tf.reduce_sum(masked_tnqs, axis=1)

        non_terminals = 1 - tf.cast(terminals, tf.float32)
        target = (
            rewards + non_terminals * (self.discount ** num_steps) * next_actions_tnqs
        )

        with tf.GradientTape() as tape:
            current_qs = self.online_model(current_states)
            masked_current_qs = current_qs * tf.one_hot(actions, self.action_space)
            pred = tf.reduce_sum(masked_current_qs, axis=1)

            loss = self.loss_fn(target, pred)

        gradients = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.online_model.trainable_variables)
        )
