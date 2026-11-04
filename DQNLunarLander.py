import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import gymnasium as gym
from collections import deque
from collections import namedtuple
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent:
    def __init__(self, env, discount_factor=0.95,
                 epsilon_greedy=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=1e-3,
                 max_memory_size=10000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        #Policy Network
        self.model = nn.Sequential(nn.Linear(self.state_size, 200), nn.ReLU(), nn.Linear(200, 100), nn.ReLU(),
                                   nn.Linear(100, self.action_size))
        #Target Network
        self.model_clone = nn.Sequential(nn.Linear(self.state_size, 200), nn.ReLU(), nn.Linear(200, 100), nn.ReLU(),
                                         nn.Linear(100, self.action_size))

        #Initialize target network with the same weights
        self.update_target_network()

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def update_target_network(self):
        #Copies weights from the policy network to the target network.
        self.model_clone.load_state_dict(self.model.state_dict())

    def remember(self, transition):
        self.memory.append(transition)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0) #From dimension (4,) to (1,4)
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def _learn(self, batch_samples):
        transitions = batch_samples
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_batch = torch.FloatTensor(np.array(states))
        actions_batch = torch.LongTensor(actions).unsqueeze(1)
        next_states_batch = torch.FloatTensor(np.array(next_states))
        rewards_batch = torch.FloatTensor(rewards).unsqueeze(1)
        dones_batch = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            #Target Net to find target values stabilizes training
            q_target = self.model_clone(next_states_batch).max(dim=1, keepdim=True)[0]
            target = rewards_batch + self.gamma * q_target * (1 - dones_batch)

        #Policy Net so the optimizer can update its parameters
        pred = self.model(states_batch).gather(1, actions_batch)

        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        loss.backward()
        self.optimizer.step()

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        return self._learn(samples)

def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    episodes = np.arange(len(history)) + 1
    plt.plot(episodes, history, lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Total rewards', size=20)
    plt.show()


n_episodes = 500
batch_size = 32
init_replay_memory_size = 1000

if __name__ == '__main__':
    env = gym.make('LunarLander-v3')
    agent = DQNAgent(env)
    state, _ = env.reset()

    #Filling up the replay-memory
    for i in range(init_replay_memory_size):
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        terminated = done or truncated
        agent.remember(Transition(state, action, reward, next_state, terminated))
        if terminated:
            state, _ = env.reset()
        else:
            state = next_state

    total_rewards = []
    for e in range(n_episodes):
        state, _ = env.reset()
        score = 0
        finished = False
        while not finished:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            finished = done
            agent.remember(Transition(state, action, reward, next_state, terminated))
            state = next_state
            score += reward

            agent.replay(batch_size)

            if terminated:
                total_rewards.append(score)
                print(f'Episode: {e}/{n_episodes}, Total reward: {score}, Epsilon: {agent.epsilon:.3f}')
                break
        #Sync target network weights at the end of every episode
        agent.update_target_network()

    plot_learning_history(total_rewards)