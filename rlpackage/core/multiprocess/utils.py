import torch
import gym

from rlpackage.core.replayBuffer.replayBuffer import ArrayReplayBuffer
from rlpackage.core.environment.env import EnvInfo
from rlpackage.core.episode.episodeExperience import EpisodeExperience
from rlpackage.core.policy.policy import DQN

from multiprocessing import Process, Manager, Queue

from typing import List, Union

def collectExperience(policy:DQN, env_name:str, queue:Queue) -> None:
    """Collect experience from the environment

    Args:
        policy (DQN): the policy used to act
        env (gym.Env): the Environment linked to the process
        queue (Queue): the queue to save the experience
    """
    env = gym.make(env_name, render_mode=None)
    env_info = EnvInfo.from_env(env)
    episode = EpisodeExperience(env_info)
    obs, info = env.reset(seed=42)
    
    step = 0
    while True:
        if queue.qsize() <= 100: # We don't want to run for nothing and to give old updates
            action = policy.act(obs)
            next_obs, reward, done, timelimit, info = env.step(action)
            episode.append(obs, action, reward, done or timelimit, next_obs)
            
            if done or timelimit:
                obs, info = env.reset()
                episode.to_numpy()
                queue.put(episode)
                episode = EpisodeExperience(env_info)

            obs = next_obs #Do not forget to switch
        
            
def storeExperience(replay_buffer: ArrayReplayBuffer, queues: List[Queue]):
    """Store experiences from the queues

    Args:
        replay_buffer (ArrayReplayBuffer): The replay buffer to store the info
        queues (List[Queue]): the queues that contain the episode
    """
    for q in queues:
        if not q.empty():
            episode = q.get_nowait()
            replay_buffer.store_episode(episode)