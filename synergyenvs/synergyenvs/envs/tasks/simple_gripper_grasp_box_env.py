from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from synergyenvs.envs.robots.simple_gripper_grasp_box import SimpleGripperGraspBox
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np

class SimpleGripperGraspBoxEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = SimpleGripperGraspBox()
        BaseBulletEnv.__init__(self, self.robot)
        self.camera = Camera(self)
        self.reward_type = 'continuous'
        self.distance_threshold = 0.05

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0020, frame_skip=5)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.to_target_vec

        # reward = - 100 * np.linalg.norm(self.robot.object.pose().xyz() - [1, 0.87, 0.4])
        reward = self.compute_reward(state["achieved_goal"], state["desired_goal"], {})
        self.HUD(state, a, True)
        return state, reward, False, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(achieved_goal - desired_goal, ord=2)
        if self.reward_type == 'sparse':
            return -(dist > self.distance_threshold).astype(np.float32)
        else:
            return -dist

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
