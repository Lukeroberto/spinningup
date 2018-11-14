import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.dqn.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class DQNBuffer:
    """
    A buffer for storing DQN trajectories
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        """
        obs_dim: dimension of the observation (s)
        act_dim: dimension of the action (a)
        size: max size of the buffer
        gamma: discount factor
        """

        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_buf), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, adv, rew, val, logp):
        """
        Append one timestep of agent environment interaction to the buffer
        """
        assert self.ptr < self.max_size # buffer needs enough space

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.adv_buf[self.ptr] = adv
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def get_tuple(index):
        return (self.obs_buf[index], self.act_buf[index], self.adv_buf[index], self.rew_buf[index])

    def sample_batch(self, batch_size):
        batch_ind = np.random.choice(self.ptr, batch_size, replace=False)
        return [get_tuple(i) for i in batch_ind]
        
