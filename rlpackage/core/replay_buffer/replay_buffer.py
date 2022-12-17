"""Define replaybuffer class"""
from typing import Dict, Tuple, Union
import numpy as np

from core.environment.env import EnvInfo
from core.episode.episodeExperience import EpisodeExperience

class Sample():
    """Base class for a sample"""
    def __init__(self, replay_buffer, indicies: Union[np.ndarray, Tuple[int, int]]):
        if isinstance(indicies, np.ndarray):
            self.obs = replay_buffer.obs_arr[indicies]
            self.act = replay_buffer.act_arr[indicies]
            self.rew = replay_buffer.rew_arr[indicies]
            self.done = replay_buffer.done_arr[indicies]
            self.next_obs = replay_buffer.next_obs_arr[indicies]

            if replay_buffer.act_log_prob is not None:
                self.act_log_prob = replay_buffer.act_log_prob[indicies]
        elif isinstance(indicies, tuple):
            self.obs = replay_buffer.obs_arr[indicies[0]: indicies[1]]
            self.act = replay_buffer.act_arr[indicies[0]: indicies[1]]
            self.rew = replay_buffer.rew_arr[indicies[0]: indicies[1]]
            self.done = replay_buffer.done_arr[indicies[0]: indicies[1]]
            self.next_obs = replay_buffer.next_obs_arr[indicies[0]: indicies[1]]

            if replay_buffer.act_log_prob is not None:
                self.act_log_prob = replay_buffer.act_log_prob[indicies[0]: indicies[1]]
        else:
            raise TypeError("The indicies for the samples do not have the good type.")


class ArrayReplayBuffer():
    """
    ArrayReplayBuffer
    """
    def __init__(self, env_info:EnvInfo, max_size:int=100000, store_trajectory=True):
        self.env_info = env_info
        self.obs_arr = np.zeros((max_size, ) + self.env_info.obs_dim)
        self.act_arr = np.zeros((max_size, ) + self.env_info.act_dim)
        self.rew_arr = np.zeros((max_size, ) + self.env_info.rew_dim)
        self.done_arr = np.zeros((max_size, ))
        self.next_obs_arr = np.zeros((max_size, ) + self.env_info.obs_dim)

        self.store_trajectory = store_trajectory
        if store_trajectory:
            self.traj_ind_arr = -np.ones((max_size,), dtype=np.int32)
            self.n_traj = 0
            self.traj_dict = {}


        self.act_log_prob = None


        self.size = 0
        self.idx = 0
        self.max_size = max_size
    def update_traj(self, next_idx:int):
        """Update the array and the dict that keeps track of the trajectories"""
        min_traj_ind, max_traj_ind = self.traj_ind_arr[self.idx], self.traj_ind_arr[next_idx]
        for traj_ind in range(min_traj_ind, max_traj_ind+1):
            if traj_ind in self.traj_dict:
                del self.traj_dict[traj_ind]
        if next_idx >= self.idx:
            self.traj_ind_arr[self.idx: next_idx] = self.n_traj
        else:
            self.traj_ind_arr[self.idx: len(self.max_size)] = self.n_traj
            self.traj_ind_arr[:next_idx] = self.n_traj
        self.traj_dict[self.n_traj] = (self.idx, next_idx)
        self.n_traj += 1

    def store_episode(self, episode:EpisodeExperience):
        """Store the information of an episode inside the replay_buffer"""
        episode.to_numpy()
        if self.idx + len(episode.obs_arr)>self.max_size:
            next_idx = len(episode.obs_arr) - (self.max_size-self.idx)
            if self.store_trajectory:
                self.update_traj(next_idx)
            self.obs_arr[self.idx: self.max_size] = episode.obs_arr[:self.max_size-self.idx]
            self.act_arr[self.idx: self.max_size] = episode.act_arr[:self.max_size-self.idx]
            self.rew_arr[self.idx: self.max_size] = episode.rew_arr[:self.max_size-self.idx]
            self.done_arr[self.idx: self.max_size] = episode.done_arr[:self.max_size-self.idx]
            self.next_obs_arr[self.idx: self.max_size] = episode.next_obs_arr[:self.max_size-self.idx]

            self.obs_arr[:next_idx] = episode.obs_arr[self.max_size-self.idx:]
            self.act_arr[:next_idx] = episode.act_arr[self.max_size-self.idx:]
            self.rew_arr[:next_idx] = episode.rew_arr[self.max_size-self.idx:]
            self.done_arr[:next_idx] = episode.done_arr[self.max_size-self.idx:]
            self.next_obs_arr[:next_idx] = episode.next_obs_arr[self.max_size-self.idx:]

            if len(episode.act_log_prob)>0:
                if self.act_log_prob is None:
                    self.act_log_prob = np.zeros((self.max_size, ) + self.env_info.act_dim)

                self.act_log_prob[self.idx: self.max_size] = episode.act_log_prob[:self.max_size-self.idx]
                self.act_log_prob[:next_idx] = episode.act_log_prob[self.max_size-self.idx:]

            self.size = self.max_size
            self.idx = next_idx

        else:
            next_idx = self.idx + len(episode.obs_arr)
            if self.store_trajectory:
                self.update_traj(next_idx)

            self.obs_arr[self.idx: next_idx] = episode.obs_arr
            self.act_arr[self.idx: next_idx] = episode.act_arr
            self.rew_arr[self.idx: next_idx] = episode.rew_arr
            self.done_arr[self.idx: next_idx] = episode.done_arr
            self.next_obs_arr[self.idx: next_idx] = episode.next_obs_arr

            if len(episode.act_log_prob)>0:
                if self.act_log_prob is None:
                    self.act_log_prob = np.zeros((self.max_size, ) + self.env_info.act_dim)

                self.act_log_prob[self.idx: next_idx] = episode.act_log_prob


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

    def sample_trajectories(self, sample_size:int, horizon:int=None) -> Sample:
        """Sample trajectories"""
        #TODO: test time to execute when horizon is fixed
        nb_sample = max(sample_size, len(self.traj_dict.keys()))
        traj_ind = np.array(list(self.traj_dict.values()))
        indicies = np.arange(len(traj_ind))
        np.random.shuffle(indicies)
        traj_ind = traj_ind[indicies[:nb_sample]]
        if horizon is None:
            return [Sample(self, (ind[0], ind[1])) for ind in traj_ind]
        else:
            return Sample(self, np.apply_along_axis(lambda ind:np.arange(ind[0], ind[0]+horizon), 1, traj_ind))

    def reset(self):
        """Reset replay buffer"""
        self.size = 0
        self.idx = 0
        if self.store_trajectory:
            self.traj_ind_arr = -np.ones((self.max_size,), dtype=np.int32)
            self.n_traj = 0
            self.traj_dict = {}
