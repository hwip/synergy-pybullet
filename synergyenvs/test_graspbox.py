import gym
import time

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import synergyenvs

env = gym.make("GraspBoxPybullet-v0")
env.render()
o = env.reset()

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

for _ in range(1000):
    env.render()
    env.camera_adjust()
    # action, _states = model.predict(o)
    action = env.action_space.sample()
    o, r, done, info = env.step(action)
    print(o, r, done, info)
    if done:
      o = env.reset()
    time.sleep(0.2)

env.close()
