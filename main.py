import logging

import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d import LunarLander
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

from LunarLander import Params, LunarLanderAgent
from qlearning import QLearning, EpsilonGreedy

if __name__ == '__main__':
    params = Params(
        total_episodes=1000,
        episode_milestone_interval=250,
        learning_rate=0.25,
        discount_factor=0.65,
        initial_epsilon=0.25,
        epsilon_decay_rate=0.95,
        final_epsilon=0.1,
        seed=42,
        n_runs=1,
        action_size=0,
        state_size=0,
        bin_size=10,
        bin_count=(10, 10, 10, 10, 10, 10, 2, 2)
    )
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    env = gym.make(
        'LunarLander-v3',
        continuous=False,
        gravity=-9.8,
        enable_wind=False,
        wind_power=5,
        turbulence_power=1.5,
        render_mode='rgb_array'
    )
    env = RecordVideo(
        env,
        video_folder='./videos/',
        name_prefix='training',
        episode_trigger=lambda x: x % params.episode_milestone_interval == 0,
    )
    env = RecordEpisodeStatistics(env, buffer_length=params.total_episodes)

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=int(np.prod(params.bin_count)))

    env.action_space.seed(params.seed)

    learner = QLearning(
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        initial_epsilon=params.initial_epsilon,
        final_epsilon=params.final_epsilon,
        epsilon_decay_rate=params.epsilon_decay_rate,
        seed=params.seed,
    )

    agent = LunarLanderAgent(
        env=env,
        params=params,
        explorer=explorer,
        learner=learner,
    )

    rewards, steps, episodes, qtables, all_states, all_actions = agent.run_env()

    res, st = agent.postprocess(episodes, params, rewards, steps)
    qtable = qtables.mean(axis=0)

    agent.plot_actions_states_distribution(states=all_states, actions=all_actions)
    agent.plot_learning_progress()

    env.close()

