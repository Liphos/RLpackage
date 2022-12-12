"""Test the environment with sb3"""
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env

def test_environment(env_name:str):
    """Test the environment with stable_baselines3"""
    # Parallel environments
    test_env = make_vec_env(env_name, n_envs=4)

    model = stable_baselines3.PPO("MlpPolicy", test_env, verbose=1)
    model.learn(total_timesteps=250000)
