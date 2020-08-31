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

        A = [[1, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]

        C = [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]]

        x0, y0, z0 = 0, 0, 0
        vx0, vy0, vz0 = 0, 0, 0

        dim_states = len(A[0])
        self.P = 1.0e-3 * np.eye(dim_states)
        Q = 1.0e-4 * np.eye(dim_states)  # process noise covariance

        dim_measurements = len(C)
        R = 1.0e-1 * np.eye(dim_measurements)  # measurement noise covariance

        self.x = [x0, y0, z0, vx0, vy0, vz0]
        self.kf = pykalman.KalmanFilter(transition_matrices=A,
                                        observation_matrices=C,
                                        initial_state_mean=self.x,
                                        initial_state_covariance=self.P,
                                        transition_covariance=Q,
                                        observation_covariance=R)

    def filter(self, measurement: List):
        self.x, self.P = self.kf.filter_update(self.x, self.P, measurement)


        return self.x, self.P
