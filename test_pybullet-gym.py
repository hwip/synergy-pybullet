import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time

env = gym.make('HumanoidPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
o = env.reset()  # should return a state vector if everything worked
for _ in range(1000):
  env.render()
  a = env.action_space.sample()
  o, r, done, info = env.step(a)
  if done:
    o = env.reset()
  time.sleep(0.2)
env.close()

