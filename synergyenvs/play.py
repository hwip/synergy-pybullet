import gym
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, HER, DDPG

import synergyenvs

env = gym.make("GraspBoxPybullet-v0")
env.render()
o = env.reset()

# model = PPO2(MlpPolicy, env, verbose=1)
model = HER('MlpPolicy', env, DDPG, n_sampled_goal=4, verbose=0)
model.load("./her_graspbox-1")

env.camera_adjust()

for _ in range(10000):
    env.render()
    action, _states = model.predict(o)
    # action = env.action_space.sample()
    o, r, done, info = env.step(action)
    print(o, r, done, info)
    if done:
      o = env.reset()
    time.sleep(0.1)

env.close()