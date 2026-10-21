import numpy as np


# https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/#sphx-glr-tutorials-training-agents-frozenlake-q-learning-py
class QLearning:
    def __init__(
            self,
            learning_rate: float,
            discount_factor: float,
            state_size: int,
            action_size: int):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor # AKA Gamma
        self.state_size = state_size
        self.action_size = action_size
        self.reset_qtable()

    def update(self, state, action, reward, next_state, terminated=False):
        next_q = (not terminated) * np.max(self.qtable[next_state])
        delta = (
            reward
            + self.discount_factor * next_q
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, initial_epsilon: float, final_epsilon: float, epsilon_decay_rate: float, seed=None):
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.rng = np.random.default_rng(seed)

    def choose_action(self, action_space, state, qtable):
        if self.rng.random() < self.epsilon:
            action = action_space.sample()

        else:
            #action = np.argmax(qtable[state])
            #max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            best_action = np.flatnonzero(
                qtable[state, :] == np.max(qtable[state, :])
            )
            action = self.rng.choice(best_action)

        return int(action)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay_rate)