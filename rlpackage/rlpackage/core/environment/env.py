import gym
from gym.spaces import Box

class EnvInfo():
    """
    To store the information about the environment
    """
    def __init__(self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box, reward_dim: int, async_env=True, n_envs=1):
        if not async_env:
            self.observation_space = observation_space
            self.action_space = action_space
            
        else:
            self.observation_space = Box(high=observation_space.high[0], low=observation_space.low[0], shape=observation_space.shape[1:])
            if type(action_space)==gym.spaces.Discrete:
                self.action_space = action_space[0]
            elif type(action_space)==gym.spaces.Box:
                self.action_space = Box(high=action_space.high[0], low=action_space.low[0], shape=action_space.shape[1:])
            else:
                raise TypeError("action_space don't have a supported type")
            
            self.obs_dim = observation_space.shape[1:]            
        
        self.act_dim = () if type(self.action_space)==gym.spaces.Discrete else self.action_space.shape
        self.obs_dim = self.observation_space.shape
        
        self.rew_dim = () if reward_dim == 1 else (reward_dim,) 
        self.async_env = async_env
        self.n_envs = n_envs
        
    @staticmethod
    def from_env(env:gym.Env, async_env=True, n_envs=1, rew_dim:int=1):
        # Create envInfo object from env
        return EnvInfo(env.action_space, env.observation_space, rew_dim, async_env=async_env, n_envs=n_envs)