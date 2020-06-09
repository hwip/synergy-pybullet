from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from synergyenvs.envs.robots.grasp_box import GraspBox
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np


class GraspBoxEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = GraspBox()
        BaseBulletEnv.__init__(self, self.robot)
        self.camera = Camera(self)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0020, frame_skip=5)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.to_target_vec

        # reward = - 100 * np.linalg.norm(self.robot.object.pose().xyz() - [1, 0.87, 0.4])
        reward = self.compute_reward(state["achieved_goal"], state["desired_goal"], {})

        self.HUD(state, a, False)
        return state, reward, False, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return -10*np.linalg.norm(achieved_goal-desired_goal, ord=2)

    def camera_adjust(self):
        yaw = 10
        self.camera.move_and_look_at(50, 1, 1, 1.5, 0.5, 0.5)


class Camera():
    def __init__(self, env):
        self.env = env
        pass

    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        distance = 0.5
        yaw = 5
        self.env._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)
