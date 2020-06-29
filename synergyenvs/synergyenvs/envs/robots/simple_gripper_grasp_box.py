from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot
import pybullet
import gym
from gym import error, spaces
import numpy as np
import os, inspect

class SimpleGripperGraspBox(URDFBasedRobot):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        URDFBasedRobot.__init__(self, "/root/synergyenvs/synergyenvs/envs/assets/hand/simple_gripper.xml",
                                "simple_gripper", action_dim=3, obs_dim=27, self_collision=False)
        self.action_space = gym.spaces.Box(-np.ones([3]), np.ones([3]), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(21,), dtype='float32'),
        ))

        # ranges of each joint in the hand, described in grasp_block.xml
        self.joints = []
        self.ctrlrange = []
        self.jname = ["table_to_base", "base_to_right_finger", "base_to_left_finger"]
        self.control_method = 'position'
        self.object_loading = False
        self.gripper_loading = False

    def reset(self, bullet_client):
        self._p = bullet_client
        self.ordered_joints = []

        full_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", self.model_urdf)
        if not self.gripper_loading:
            self.gripper_loading = True
            if self.self_collision:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                                                                                               self._p.loadURDF(
                                                                                                   full_path,
                                                                                                   basePosition=self.basePosition,
                                                                                                   baseOrientation=self.baseOrientation,
                                                                                                   useFixedBase=self.fixed_base,
                                                                                                   flags=pybullet.URDF_USE_SELF_COLLISION))
            else:
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p,
                                                                                               self._p.loadURDF(
                                                                                                   full_path,
                                                                                                   basePosition=self.basePosition,
                                                                                                   baseOrientation=self.baseOrientation,
                                                                                                   useFixedBase=self.fixed_base))

        for j in self.jdict.values():
            j.unlock_joint()

        self.robot_specific_reset(self._p)
        self._p.setGravity(0, 0, -9.8)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()

        return s

    def robot_specific_reset(self, bullet_client):
        if not self.object_loading:
            self.object_parts, _, _, _ = self.addToScene(self._p,
                                                   self._p.loadURDF("/root/synergyenvs/synergyenvs/envs/assets/block.urdf",
                                                                    basePosition=[0, -0.4, 0.2],
                                                                    baseOrientation=[1,1,1,1],
                                                                    useFixedBase=self.fixed_base,
                                                                    flags=self._p.URDF_USE_SELF_COLLISION))
            self.object_loading = True
            # self.target_parts, _, _, _ = self.addToScene(self._p,
            #                                        self._p.loadURDF("/root/synergyenvs/synergyenvs/envs/assets/block.urdf",
            #                                                         basePosition=self.basePosition,
            #                                                         baseOrientation=self.baseOrientation,
            #                                                         useFixedBase=self.fixed_base,
            #                                                         flags=self._p.URDF_USE_SELF_COLLISION))
        # parts
        self.target_pos = np.array([0, -0.4, 0.5])
        self.object = self.object_parts["block_2_base_link"]
        # Initialize environment in each episode
        self.joints = []
        self.ctrlrange = []

        for name in self.jname:
            self.joints.append(self.jdict[name])
            self.ctrlrange.append([self.joints[-1].lowerLimit, self.joints[-1].upperLimit])
        # reset position and speed of manipulator
        # TODO: Will this work or do we have to constrain this resetting in some way?
        self.initpos = [0, 0.15, -0.15]
        self.jforce = [60, 30, 30]
        for i, joint in enumerate(self.joints):
            joint.reset_current_position(self.initpos[i], 0)

        # reset object position
        self.object.reset_orientation([1, 1, 1, 1])
        self.object.reset_position([0, -0.4, 0.2])
        self.object.reset_velocity([0, 0, 0])

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        rng = np.array(self.ctrlrange)

        actuation_range = (rng[:, 1] - rng[:, 0]) / 2
        actuation_center = (rng[:, 1] + rng[:, 0]) / 2
        nxt = actuation_center + a * actuation_range
        nxt = np.clip(nxt, rng[:, 0], rng[:, 1])
        for i, joint in enumerate(self.joints):
            joint.set_position(nxt[i], force=self.jforce[i])

    def calc_state(self):
        self.to_target_vec = self.target_pos - self.object.pose().xyz()
        return {
            "desired_goal": self.target_pos,
            "achieved_goal": self.object.pose().xyz(),
            "observation": np.concatenate([
                np.array([j.current_position() for j in self.joints]).flatten(),  # all positions
                np.array([j.current_relative_position() for j in self.joints]).flatten(),  # all speeds
                self.to_target_vec,
                self.object.pose().xyz(),
                self.target_pos])
        }
