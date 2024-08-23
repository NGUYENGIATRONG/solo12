import numpy as np
import gym
from gym import spaces
from simulation import walking_controller
import math
import random
from collections import deque
import pybullet
from simulation import pybullet_client
from utils import solo12_kinematic

import pybullet_data
# import SlopedTerrainLinearPolicy.gym_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator
from utils import get_terrain_normal as normal_estimator

policy = np.load(
    '/home/quyetnguyen/PycharmProjects/Laikago/SlopedTerrainLinearPolicy/initial_policies/initial_policy_HyQ.npy')
add_IMU_noise = False
ori_history_length = 3
ori_history_queue = deque([0] * 3 * ori_history_length,
                          maxlen=3 * ori_history_length)
support_plane_estimated_pitch = 0
support_plane_estimated_roll = 0
_pybullet_client = pybullet_client.BulletClient(connection_mode=pybullet.GUI)
model_path = '/home/quyetnguyen/PycharmProjects/Laikago/simulation/robots/solo12/solo12.urdf'
INIT_POSITION = [0, 0, 0.65]
INIT_ORIENTATION = [0, 0, 0, 1]
solo12 = _pybullet_client.loadURDF(model_path, INIT_POSITION, INIT_ORIENTATION)


def GetObservation():
    '''
    This function returns the current observation of the environment for the interested task
    Ret:
        obs : [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
    '''
    # motor_angles = np.array(self.GetMotorAngles(), dtype=np.float32)
    # motor_velocities = np.array(self.GetMotorVelocities(), dtype=np.float32)

    pos, ori = GetBasePosAndOrientation()
    RPY = _pybullet_client.getEulerFromQuaternion(ori)
    RPY = np.round(RPY, 5)
    for val in RPY:
        if add_IMU_noise:
            val = add_noise(val)
            ori_history_queue.append(val)

    obs = np.concatenate(
        (ori_history_queue, [support_plane_estimated_roll, support_plane_estimated_pitch])).ravel()

    return obs


def GetBasePosAndOrientation():
    """
        This function returns the robot torso position(X,Y,Z) and orientation(Quaternions) in world frame
        """
    position, orientation = (_pybullet_client.getBasePositionAndOrientation(solo12))
    return position, orientation


def add_noise(sensor_value, SD=0.04):
    """
        Adds sensor noise of user defined standard deviation in current sensor_value
        """
    noise = np.random.normal(0, SD, 1)
    sensor_value = sensor_value + noise[0]
    return sensor_value


def reset():
    _theta = 0
    _last_base_position = [0, 0, 0]
    last_yaw = 0

    _pybullet_client.resetBasePositionAndOrientation(stoch2, self.INIT_POSITION, self.INIT_ORIENTATION)
    _pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])
    # reset_standing_position()

    _pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
    _n_steps = 0
    return GetObservation()


state = reset()