import gym
import time
import synergyenvs

env = gym.make("GraspBoxPybullet-v0")
env.render()
o = env.reset()

for _ in range(1000):
  env.render()
  a = env.action_space.sample()
  o, r, done, info = env.step(a)
  if done:
    o = env.reset()
  time.sleep(0.2)
env.close()