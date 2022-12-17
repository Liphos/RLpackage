"""Main file to run the algorithm"""
from typing import List
import gym
import torch
import numpy as np

from core.replay_buffer.replay_buffer import ArrayReplayBuffer
from core.environment.env import EnvInfo
from core.episode.episodeExperience import EpisodeExperience
from core.policy.policy import Policy, PPO

ENV_NAME = "LunarLander-v2"
NB_UPDT_BETWEEN_TESTS = 10
N_ENVS = 3

def test_policy(policy_alg:Policy, env_name:str, render_mode:str="human") -> int:
    """Return the reward gain in the episode when testing the policy

    Args:
        policy_alg (Policy): The policy to test
        env_name (str): the environment name
        render_mode (str): the render mode

    Returns:
        int: the cumulative reward
    """

    #env = gym.make(env_name, render_mode=render_mode)
    test_env = gym.make(env_name, render_mode=render_mode, continuous=True)
    obs, _ = test_env.reset(seed=42)
    cumulative_reward = 0
    done = False
    while not done:
        actions = policy_alg.act(obs, deterministic=True)
        action_clip = np.clip(actions, test_env.action_space.low, test_env.action_space.high)
        next_obs, reward, done, timelimit, _ = test_env.step(action_clip)
        cumulative_reward += reward
        done = done or timelimit

        obs = next_obs

    test_env.close()


    return cumulative_reward

def reset_envs(envs:gym.vector.AsyncVectorEnv):
    """Reset the environment"""

    obs, info = envs.reset()
    return obs, info

def reset_buffers(replay_buffer:ArrayReplayBuffer,
                  episodes:List[EpisodeExperience],
                  env_info:EnvInfo):
    """Reset the replay buffer"""

    replay_buffer.reset()
    for incr in enumerate(episodes):
        episodes[incr[0]] = EpisodeExperience(env_info)

def train_loop():
    """The training loop"""

    env = gym.vector.AsyncVectorEnv([lambda: gym.make(ENV_NAME, render_mode=None, continuous=True)
                                     for _ in range(N_ENVS)])
    env_info = EnvInfo.from_env(env, async_env=True, n_envs=N_ENVS)
    replay_buffer = ArrayReplayBuffer(env_info)
    episodes = [EpisodeExperience(env_info) for _ in range(N_ENVS)]
    policy = PPO(env_info, replay_buffer, optimizer=torch.optim.Adam, batch_size=1)

    obs, _ = env.reset()

    mean_loss = 0
    for _ in range(2000000):
        #Collect experience from the environment
        action, action_log_prob = policy.act(obs)
        action_clip = np.clip(action, env_info.action_space.low, env_info.action_space.high)
        next_obs, reward, done, timelimit, _ = env.step(action_clip)
        if np.any(np.isnan(action_clip)) or np.any(np.isnan(action_log_prob)):
            print()
        for incr in range(N_ENVS):
            episodes[incr].append(obs[incr],
                                  action[incr],
                                  reward[incr],
                                  done[incr] or timelimit[incr],
                                  next_obs[incr],
                                  action_log_prob[incr])

            if done[incr] or timelimit[incr]:
                episodes[incr].to_numpy()
                replay_buffer.store_episode(episodes[incr])
                episodes[incr] = EpisodeExperience(env_info)

        obs = next_obs #Do not forget to switch

        loss, reset_info = policy.train()

        if "buffers" in reset_info:
            reset_buffers(replay_buffer, episodes, env_info)
        if "envs" in reset_info:
            obs, _ = reset_envs(env)

        if loss is not None:
            mean_loss += loss
            if policy.train_step % NB_UPDT_BETWEEN_TESTS == 0:
                mean_loss = mean_loss / NB_UPDT_BETWEEN_TESTS
                test_rew = test_policy(policy, ENV_NAME, render_mode=None)
                print(f"loss : {mean_loss:3.6f}  | "
                      f"reward: {test_rew:.6f}  | "
                      f"replay_buffer_size: {replay_buffer.size:8f}")
                mean_loss = 0
    env.close()


if __name__ == '__main__':
    print("=======================================================================================")

    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("=======================================================================================")

    train_loop()

