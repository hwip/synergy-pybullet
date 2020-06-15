from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import gym
from gym import error, spaces
import numpy as np


class GraspBox(MJCFBasedRobot):
    min_target_placement_radius = 0.1
    max_target_placement_radius = 0.8
    min_object_placement_radius = 0.1
    max_object_placement_radius = 0.8

    def __init__(self):
        MJCFBasedRobot.__init__(self, "/root/synergyenvs/synergyenvs/envs/assets/hand/grasp_block.xml", "body0",
                                action_dim=21, obs_dim=102)
        self.action_space = gym.spaces.Box(-np.ones([21]), np.ones([21]), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(96,), dtype='float32'),
        ))
        self.jname = ["robot0:slider",
                      "robot0:WRJ1", "robot0:WRJ0",
                      "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1",
                      "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:LFJ4",
                      "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1",
                      "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1",
                      "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
        self.djname = ["robot0:FFJ0", "robot0:MFJ0", "robot0:RFJ0", "robot0:LFJ0"]
        # ranges of each joint in the hand, described in grasp_block.xml
        self.ctrlrange = np.array([[-0.1, 0.4],
                          [-0.489, 0.14], [-0.698, 0.489],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571], [0, 0.785],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571],
                          [-1.047, 1.047], [0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0]])
        self.forcerange = np.array([[-1.0, 1.0],
                           [-4.784, 4.785], [-2.175, 2.174],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245], [-0.9, 0.9],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245],
                           [-2.3722, 2.3722], [-1.45, 1.45], [-0.99, 0.99], [-0.99, 0.99], [-0.81, 0.81]])

    def robot_specific_reset(self, bullet_client):
        # parts
        self.fingertip = self.parts["robot0:vertical slider"]
        self.target = self.parts["target"]
        self.object = self.parts["object"]

        self.joints = []
        for name in self.jname:
            self.joints.append(self.jdict[name])

        self.distal_joints = []
        for name in self.djname:
            self.distal_joints.append(self.jdict[name])

        self._object_hit_ground = False
        self._object_hit_location = None

        # reset position and speed of manipulator
        # TODO: Will this work or do we have to constrain this resetting in some way?
        for joint in self.joints:
            joint.reset_current_position(0, 0)
        self.joints[0].reset_current_position(0, 0)

        self.zero_offset = np.array([0.45, 0.55, 0])

        # reset object position
        self.object.reset_orientation([1,1,1,1])
        self.object.reset_position([1, 0.85, 0.205])
        self.object.reset_velocity([0, 0, 0])

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        actuation_range = (self.ctrlrange[:, 1] - self.ctrlrange[:, 0]) / 2
        actuation_center = (self.ctrlrange[:, 1] + self.ctrlrange[:, 0]) / 2
        nxt = actuation_center + a * actuation_range
        nxt = np.clip(nxt, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        for i, joint in enumerate(self.joints):
            pass
            # joint.set_motor_torque(0.01 * nxt[i])
            joint.set_position(nxt[i])
        self.distal_joints[0].set_position(nxt[5])
        self.distal_joints[1].set_position(nxt[8])
        self.distal_joints[2].set_position(nxt[12])
        self.distal_joints[3].set_position(nxt[15])

    def calc_state(self):
        self.to_target_vec = self.target.pose().xyz() - self.object.pose().xyz()
        return {
            "desired_goal": self.target.pose().xyz(),
            "achieved_goal": self.object.pose().xyz(),
            "observation": np.concatenate([
                np.array([j.current_position() for j in self.joints]).flatten(),  # all positions
                np.array([j.current_relative_position() for j in self.joints]).flatten(),  # all speeds
                self.to_target_vec,
                self.fingertip.pose().xyz(),
                self.object.pose().xyz(),
                self.target.pose().xyz()])
        }
