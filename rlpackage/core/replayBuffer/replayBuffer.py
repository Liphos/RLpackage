import numpy as np

from rlpackage.core.environment.env import EnvInfo
from rlpackage.core.episode.episodeExperience import EpisodeExperience

class Sample():
    def __init__(self, replay_buffer, indicies: np.ndarray):
        self.obs = replay_buffer.obs_arr[indicies]
        self.act = replay_buffer.act_arr[indicies]
        self.rew = replay_buffer.rew_arr[indicies]
        self.done = replay_buffer.done_arr[indicies]
        self.next_obs = replay_buffer.next_obs_arr[indicies]
        
        if replay_buffer.act_log_prob is not None:
            self.act_log_prob = replay_buffer.act_log_prob[indicies]

class ArrayReplayBuffer():
    """
    ArrayReplayBuffer
    """
    def __init__(self, env_info:EnvInfo, max_size:int=100000):
        self.env_info = env_info
        self.obs_arr = np.zeros((max_size, ) + self.env_info.obs_dim)
        self.act_arr = np.zeros((max_size, ) + self.env_info.act_dim)
        self.rew_arr = np.zeros((max_size, ) + self.env_info.rew_dim)
        self.done_arr = np.zeros((max_size, ))
        self.next_obs_arr = np.zeros((max_size, ) + self.env_info.obs_dim)
        
        self.act_log_prob = None
        
        self.size = 0
        self.idx = 0
        self.max_size = max_size
        
    def store_episode(self, episode:EpisodeExperience):
        episode.to_numpy()
        
        if self.idx + len(episode.obs_arr)>self.max_size:
            self.obs_arr[self.idx: self.max_size] = episode.obs_arr[:self.max_size-self.idx]
            self.act_arr[self.idx: self.max_size] = episode.act_arr[:self.max_size-self.idx]
            self.rew_arr[self.idx: self.max_size] = episode.rew_arr[:self.max_size-self.idx]
            self.done_arr[self.idx: self.max_size] = episode.done_arr[:self.max_size-self.idx]
            self.next_obs_arr[self.idx: self.max_size] = episode.next_obs_arr[:self.max_size-self.idx]
            
            self.obs_arr[:len(episode.obs_arr) - (self.max_size-self.idx)] = episode.obs_arr[self.max_size-self.idx:]
            self.act_arr[:len(episode.obs_arr) - (self.max_size-self.idx)] = episode.act_arr[self.max_size-self.idx:]
            self.rew_arr[:len(episode.obs_arr) - (self.max_size-self.idx)] = episode.rew_arr[self.max_size-self.idx:]
            self.done_arr[:len(episode.obs_arr) - (self.max_size-self.idx)] = episode.done_arr[self.max_size-self.idx:]
            self.next_obs_arr[:len(episode.obs_arr) - (self.max_size-self.idx)] = episode.next_obs_arr[self.max_size-self.idx:]
            
            if len(episode.act_log_prob)>0:
                if self.act_log_prob == None:
                    self.act_log_prob = np.zeros((self.max_size, ) + self.env_info.act_dim)
                    
                self.act_log_prob[self.idx: self.max_size] = episode.act_log_prob[:self.max_size-self.idx]
                self.act_log_prob[:len(episode.obs_arr) - (self.max_size-self.idx)] = episode.act_log_prob[self.max_size-self.idx:]
            
            self.size = self.max_size
            self.idx = len(episode.obs_arr) - (self.max_size-self.idx)
            
        else:
            self.obs_arr[self.idx: self.idx + len(episode.obs_arr)] = episode.obs_arr
            self.act_arr[self.idx: self.idx + len(episode.act_arr)] = episode.act_arr
            self.rew_arr[self.idx: self.idx + len(episode.rew_arr)] = episode.rew_arr
            self.done_arr[self.idx: self.idx + len(episode.done_arr)] = episode.done_arr
            self.next_obs_arr[self.idx: self.idx + len(episode.next_obs_arr)] = episode.next_obs_arr
            
            if len(episode.act_log_prob)>0:
                if self.act_log_prob is None:
                    self.act_log_prob = np.zeros((self.max_size, ) + self.env_info.act_dim)
                    
                self.act_log_prob[self.idx: self.idx + len(episode.obs_arr)] = episode.act_log_prob
                
                
            self.size += len(episode.obs_arr)
            self.idx += len(episode.obs_arr)
            
            self.size = np.minimum(self.size, self.max_size)
            self.idx = self.idx % self.max_size
            
    def sample(self, sample_size:int) -> Sample:
        """
        Return a random sample of the replay_buffer

        Args:
            sample_size (int): the size of the sample 
        Returns:
            Sample: A random sample of the replay_buffer
        """
        indicies = np.arange(self.size)
        np.random.shuffle(indicies)
        indicies = indicies[:sample_size]
        return Sample(self, indicies)
    
    def reset(self):
        """Reset replay buffer
        """
        self.size = 0
        self.idx = 0
        