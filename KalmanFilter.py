"""
Code inspired and used from
https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Tracking.html#kalman-filter
"""

from typing import List

import numpy as np
import pykalman


class KalmanFilter(object):
    def __init__(self) -> None:
        super().__init__()

        # todo: define state transition matrix for 3D positions and velocities
        A = []

        # todo: define observation matrix for 3D position measurements
        C = []


        # todo: set covariance matrices (use small values)
        self.P = None
        Q = None  # process noise covariance
        R = None  # measurement noise covariance

        # todo: initialize the state vector
        x0, y0, z0 = 0.0, 0.0, 0.0
        vx0, vy0, vz0 = 0.0, 0.0, 0.0
        self.x = [x0, y0, z0, vx0, vy0, vz0]

        #TODO: create the KF
        self.kf = pykalman.KalmanFilter(transition_matrices=None,
                                        observation_matrices=None,
                                        initial_state_mean=None,
                                        initial_state_covariance=None,
                                        transition_covariance=None,
                                        observation_covariance=None)

    def filter(self, measurement: List):
        # todo: call the filter and save the state and covariance
        # TODO: Don't forget to delete the condition
        if False:
            _, _ = self.kf.filter_update(None, None, None)

        # todo: return both x and P
        return self.x, self.P
