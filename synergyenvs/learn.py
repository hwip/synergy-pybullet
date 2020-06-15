import gym
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, HER, DDPG

import synergyenvs

env = gym.make("GraspBoxPybullet-v0")
o = env.reset()

# model = PPO2(MlpPolicy, env, verbose=1)
model = HER('MlpPolicy', env, DDPG, n_sampled_goal=4, verbose=0)
model.learn(10000)

model.save("./her_graspbox-1")

env.close()