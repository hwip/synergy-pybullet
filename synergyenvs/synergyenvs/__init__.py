from gym.envs.registration import register

def _merge(a, b):
    a.update(b)
    return a

register(
    id='GraspBoxPybullet-v0',
    entry_point='synergyenvs.envs.tasks.grasp_box_env:GraspBoxEnv',
    max_episode_steps=100,
)
