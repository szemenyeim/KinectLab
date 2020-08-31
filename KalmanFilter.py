"""Code inspired and used from https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Tracking.html#kalman-filter"""

import numpy as np
import pykalman
from typing import List

class KalmanFilter(object):
    def __init__(self) -> None:
        super().__init__()

        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        C = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        xinit = 0
        yinit = 0
        vxinit = 0
        vyinit = 0
        x0 = [xinit, yinit, vxinit, vyinit]

        P0 = 1.0e-3 * np.eye(4)
        Q = 1.0e-4 * np.eye(4)  # process noise covariance
        R = 1.0e-1 * np.eye(2)  # measurement noise covariance

        self.kf = pykalman.KalmanFilter(transition_matrices=A,
                               observation_matrices=C,
                               initial_state_mean=x0,
                               initial_state_covariance=P0,
                               transition_covariance=Q,
                               observation_covariance=R)

    def filter(self, measurement: List):
        x, P = self.kf.filter(measurement)

        return x, P
