from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import gym
import numpy as np

class GraspBox(MJCFBasedRobot):
    min_target_placement_radius = 0.1
    max_target_placement_radius = 0.8
    min_object_placement_radius = 0.1
    max_object_placement_radius = 0.8

    def __init__(self):
        MJCFBasedRobot.__init__(self, "/root/synergyenvs/synergyenvs/envs/assets/hand/grasp_block.xml", "body0", action_dim=21, obs_dim=112)
        self.action_space = gym.spaces.Box(-0.9*np.ones([21]), 0.9*np.ones([21]))

    def robot_specific_reset(self, bullet_client):
        # parts
        self.fingertip = self.parts["robot0:vertical slider"]
        self.target = self.parts["target"]
        self.object = self.parts["object"]

        self.jname = ["robot0:slider",
                      "robot0:WRJ1", "robot0:WRJ0",
                      "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1",
                      "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:LFJ4",
                      "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1",
                      "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"]
        # ranges of each joint in the hand, described in grasp_block.xml
        self.ctrlrange = [[0, 0.6],
                          [-0.489, 0.14], [-0.698, 0.489],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571], [0, 0.785],
                          [-0.349, 0.349], [0, 0.1571], [0, 0.1571],
                          [-1.047, 1.047], [0, 1.222], [-0.209, 0.209], [-0.524, 0.524], [-1.571, 0]]
        self.forcerange = [[-1.0, 1.0],
                           [-4.784, 4.785], [-2.175, 2.174],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245], [-0.9, 0.9],
                           [-0.9, 0.9], [-0.9, 0.9], [-0.7245, 0.7245],
                           [-2.3722, 2.3722], [-1.45, 1.45], [-0.99, 0.99], [-0.99, 0.99], [-0.81, 0.81]]

        self.joints = []
        for name in self.jname:
            self.joints.append(self.jdict[name])

        self._object_hit_ground = False
        self._object_hit_location = None

        # reset position and speed of manipulator
        # TODO: Will this work or do we have to constrain this resetting in some way?
        for joint in self.joints:
            joint.reset_current_position(0, 0)


        self.zero_offset = np.array([0.45, 0.55, 0])
        self.object_pos = np.concatenate([
            self.np_random.uniform(low=-1, high=1, size=1),
            self.np_random.uniform(low=-1, high=1, size=1),
            self.np_random.uniform(low=-1, high=1, size=1)
        ])

        # make length of vector between min and max_object_placement_radius
        self.object_pos = self.object_pos \
                          / np.linalg.norm(self.object_pos) \
                          * self.np_random.uniform(low=self.min_object_placement_radius,
                                                   high=self.max_object_placement_radius, size=1)

        # reset object position
        # self.parts["ball"].reset_pose(self.object_pos - self.zero_offset, np.array([0, 0, 0, 1]))

        self.target_pos = np.concatenate([
            self.np_random.uniform(low=-1, high=1, size=1),
            self.np_random.uniform(low=-1, high=1, size=1),
            self.np_random.uniform(low=-1, high=1, size=1)
        ])

        # make length of vector between min and max_target_placement_radius
        self.target_pos = self.target_pos \
                          / np.linalg.norm(self.target_pos) \
                          * self.np_random.uniform(low=self.min_target_placement_radius,
                                                   high=self.max_target_placement_radius, size=1)

        # self.parts["goal"].reset_pose(self.target_pos - self.zero_offset, np.array([0, 0, 0, 1]))

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for i, joint in enumerate(self.joints):
            joint.set_motor_torque(0.01 * float(np.clip(a[i], self.forcerange[i][0], self.forcerange[i][1])))

    def calc_state(self):
        self.to_target_vec = self.target_pos - self.object_pos
        return np.concatenate([
            np.array([j.current_position() for j in self.ordered_joints]).flatten(),  # all positions
            np.array([j.current_relative_position() for j in self.ordered_joints]).flatten(),  # all speeds
            self.to_target_vec,
            self.fingertip.pose().xyz(),
            self.object.pose().xyz(),
            self.target.pose().xyz(),
        ])