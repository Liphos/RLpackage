import numpy as np

from rlpackage.core.environment.env import EnvInfo


class EpisodeExperience():
    """
    Type to store an episode
    """
    def __init__(self, env_info:EnvInfo):
        self.obs_arr = []
        self.act_arr = []
        self.rew_arr = []
        self.done_arr = []
        self.next_obs_arr = []
        
        self.act_log_prob = []
        
    def append(self, obs, act, rew, done, next_obs, act_log_prob=None):
        self.obs_arr.append(obs)
        self.act_arr.append(act)
        self.rew_arr.append(rew)
        self.done_arr.append(done)
        self.next_obs_arr.append(next_obs)
        
        if act_log_prob is not None:
            self.act_log_prob.append(act_log_prob)
    
    def to_numpy(self):
        self.obs_arr = np.asarray(self.obs_arr, dtype=np.float32)
        self.act_arr = np.asarray(self.act_arr, dtype=np.float32)
        self.rew_arr = np.asarray(self.rew_arr, dtype=np.float32)
        self.done_arr = np.asarray(self.done_arr, dtype=np.float32)
        self.next_obs_arr = np.asarray(self.next_obs_arr, dtype=np.float32)
        
        self.act_log_prob = np.asarray(self.act_log_prob, dtype=np.float32)
        
        