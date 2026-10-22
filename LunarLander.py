from collections import defaultdict

import logging
from typing import NamedTuple

import gymnasium as gym
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from sklearn.preprocessing import KBinsDiscretizer
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from qlearning import QLearning, EpsilonGreedy

class Params(NamedTuple):
    total_episodes: int
    episode_milestone_interval: int
    learning_rate: float
    discount_factor: float
    initial_epsilon: float
    epsilon_decay_rate: float
    final_epsilon: float
    seed: int
    n_runs: int
    action_size: int
    state_size: int
    bin_size: int
    bin_count: tuple

class LunarLanderAgent:
    def __init__(
            self,
            env: gym.Env,
            params: Params,
            learner: QLearning,
            explorer: EpsilonGreedy,
    ):
        self.env = env
        self.params = params
        self.learner = learner
        self.explorer = explorer
        self.discretizer = KBinsDiscretizer(
            n_bins=params.bin_size,
            encode='ordinal',
            strategy='uniform'
        )
        self.discretizer.fit(np.array([env.observation_space.low[:6], env.observation_space.high[:6]]))

    def interval_milestone_handler(self, episode, run, info):
        if "episode" in info:
            if episode % self.params.episode_milestone_interval == 0:
                rewards = list(self.env.return_queue)
                #first_rewards = np.mean(rewards[:self.params.episode_milestone_interval])
                recent_rewards = np.mean(rewards[-self.params.episode_milestone_interval:])
                tqdm.write(f"run {run+1} -> Average reward for last {self.params.episode_milestone_interval} episodes: {recent_rewards}")


    def run_env(self):
        rewards = np.zeros((self.params.total_episodes, self.params.n_runs))
        steps = np.zeros((self.params.total_episodes, self.params.n_runs))
        episodes = np.arange(self.params.total_episodes)
        qtables = np.zeros((self.params.n_runs, self.params.state_size, self.params.action_size))
        observation, info = self.env.reset(seed=self.params.seed)
        all_states = []
        all_actions = []

        for run in range(self.params.n_runs):
            self.learner.reset_qtable()

            for episode in tqdm(
                    episodes,
                    desc=f"Run {run+1}/{self.params.n_runs} - Episodes",
                    leave=False,
                    miniters=(self.params.total_episodes/10),
                    maxinterval=30,
                    file=sys.stdout
            ):
                observation, info = self.env.reset()
                state = self.discretize_observation(observation)
                total_reward = 0
                step_count = 0
                episode_over = False

                while not episode_over:
                    action = self.explorer.choose_action(
                        action_space=self.env.action_space, state=state, qtable=self.learner.qtable
                    )

                    all_states.append(state)
                    all_actions.append(action)

                    new_observation, reward, terminated, truncated, info = self.env.step(action)
                    new_state = self.discretize_observation(new_observation)


                    episode_over = terminated or truncated

                    self.learner.qtable[state, action] = self.learner.update(
                        state, action, reward, new_state, terminated=terminated
                    )

                    total_reward += reward
                    step_count += 1
                    state = new_state

                rewards[episode, run] = total_reward
                steps[episode, run] = step_count

                self.interval_milestone_handler(episode, run, info)
                self.explorer.decay_epsilon()

            qtables[run, :, :] = self.learner.qtable

        unique_states = len(set(all_states))
        print(f"Possible states: {self.params.state_size:,}")
        print(f"Visited states: {unique_states:,}")
        print(f"State coverage: {unique_states / self.params.state_size:.2%}")

        return rewards, steps, episodes, qtables, all_states, all_actions

    def discretize_observation(self, observation):
        # Need to map continuous values to discrete values
        # Many ways of doing this, should experiment with different methods
        # Observation space is 8 features and represents the following:
        # [Index]: [Value] ([lower bound], [Upper bound])
        # 0: X Position (-2.5, 2.5)
        # 1: Y Position (-2.5, 2.5)
        # 2: X Velocity (-10, 10)
        # 3: Y Velocity (-10, 10)
        # 4: Angle (-6.28, 6.28)
        # 5: Angular Velocity (-10, 10)
        # 6: Left leg contact (0, 1) Boolean
        # 7: Right leg contact (0, 1) Boolean

        # Uniform discretization, simplest, but also VERY approximate and inefficient. Should change!!!
        bin_size = 4 # Determines how many "categories" to apply to EACH feature
        continous_observation = observation[:6].reshape(1, -1) # Remove the last two bool values

        binned_observation = self.discretizer.transform(continous_observation)

        binned_observation = np.concatenate([binned_observation.flatten(), observation[6:]])
        binned_observation = binned_observation.astype(int)
        state = np.ravel_multi_index(
            tuple(binned_observation),
            self.params.bin_count,
        )
        return int(state)

    # Visualize as a graph
    def postprocess(self, episodes, params, rewards, steps):
        results = pd.DataFrame(
            {
                "episodes": np.tile(episodes, reps=params.n_runs),
                "rewards": rewards.flatten(order="F"),
                "steps": steps.flatten(order="F"),
            }
        )
        results["cumulative_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
        st = pd.DataFrame(data={"Episodes": episodes, "steps": steps.mean(axis=1)})
        return results, st

    def plot_actions_states_distribution(self, states, actions):
        labels = {"NONE": 0, "LEFT ENGINE": 1, "MAIN ENGINE": 2, "RIGHT ENGINE": 3}

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        sns.histplot(data=states, ax=ax[0], kde=False)
        ax[0].set_title("States Distribution")
        sns.histplot(data=actions, ax=ax[1])
        ax[1].set_xticks(list(labels.values()), labels=labels.keys())
        ax[1].set_title("Actions Distribution")
        fig.tight_layout()
        plt.show()

    def plot_learning_progress(self):
        episodes = range(len(self.env.return_queue))
        rewards = list(self.env.return_queue)

        plt.figure(figsize=(15, 5))
        plt.title("Learning Progress")
        plt.plot(episodes, rewards, alpha=0.3, label="Episode Rewards")

        window = 100
        if len(rewards) > window:
            moving_average = [sum(rewards[i:i+window])/window
                              for i in range(len(rewards)-window+1)]
            plt.plot(range(window-1, len(rewards)), moving_average, label=f"{window}-Episode Moving Average", linewidth=2)

        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()




