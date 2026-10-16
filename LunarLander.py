from collections import defaultdict

import logging
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

training_period = 5 # Interval for video recording / Record every x episodes
episode_milestone_interval = 2 # Milestone interval, for additional information
num_train_episodes = 10 # number of episodes to go through

logging.basicConfig(level=logging.INFO, format='%(message)s')

env = gym.make('LunarLander-v3', continuous=False, gravity=-9.8,
               enable_wind=False, wind_power=5, turbulence_power=1.5, render_mode='rgb_array')

env = RecordVideo(
    env,
    video_folder='./videos/',
    name_prefix = 'training',
    episode_trigger=lambda x: x % training_period == 0
)
env = RecordEpisodeStatistics(env, buffer_length=num_train_episodes)

random_state = 42
observation, info = env.reset(seed=random_state)



for episode_num in range(num_train_episodes):
    observation, info = env.reset()
    episode_over = False
    total_reward = 0
    step_count = 0

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        episode_over = terminated or truncated

    if "episode" in info:
        episode_info = info["episode"]
        logging.info(f"Episode {episode_num}: "
                     f"reward={episode_info['r']:.1f}, "
                     f"length={episode_info['l']}, "
                     f"time={episode_info['t']:.2f}s")

        if episode_num % episode_milestone_interval == 0:
            recent_rewards = list(env.return_queue)[-episode_milestone_interval:]
            if recent_rewards:
                avg_reward = sum(recent_rewards)/len(recent_rewards)
                logging.info(f"-> Average reward over the last {episode_milestone_interval} episodes: {avg_reward:.2f}%")

env.close()
