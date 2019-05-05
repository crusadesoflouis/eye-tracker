from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Builtin dependencies
import math

# PyPI dependencies
import numpy as np
from filterpy.kalman import KalmanFilter


class BothEyesTracker():
    """Both Eyes Tracker using a Kalman filter

    Parameters
    ----------
    x_init : np.array
        Initial states of the center of the eyes, distance between eyes, and angle of eyes vector
        [x, vx, y, vy, size, distance, rads]

    """

    def __init__(self, x_init):

    # SETTINGS

        # Prediction Errors
        axdev = 10 # pixels/s^2
        aydev = 10 # pixels/s^2
        sdev = 5 # pixels
        ddev = 1 # pixels
        alfadev = 0.5 # rads

        # Measurement Errors
        xdev_measure = 2
        ydev_measure = 2
        sdev_measure = 0.5
        ddev_measure = 1
        alfadev_measure = 0.1

        # Initial Errors of States
        xdev_init = 240
        ydev_init = 320
        sdev_init = 10
        ddev_init = 120
        alfadev_init = 0.5
        # FPS
        fps = 30

    # KALMAN

        # Initialize the filter's matrices
        self.my_filter = KalmanFilter(dim_x=7, dim_z=5)

        self.my_filter.F = np.array([[1, 1/float(fps), 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1/float(fps), 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]
                                ])  # state transition matrix

        self.my_filter.Q = np.array([[np.power(axdev, 2)*np.power(1/float(fps), 4)/36, np.power(axdev, 2)*np.power(1/float(fps),3)/12, 0, 0, 0, 0, 0],
                                [np.power(axdev, 2)*np.power(1/float(fps), 3)/12, np.power(axdev, 2)*np.power(1/float(fps), 2)/4, 0, 0, 0, 0, 0],
                                [0, 0, np.power(aydev, 2)*np.power(1/float(fps), 4)/36, np.power(aydev, 2)*np.power(1/float(fps),3)/12, 0, 0, 0],
                                [0, 0, np.power(aydev, 2)*np.power(1/float(fps), 3)/12, np.power(aydev, 2)*np.power(1/float(fps), 2)/4, 0, 0, 0],
                                [0, 0, 0, 0, np.power(sdev, 2), 0, 0],
                                [0, 0, 0, 0, 0, np.power(ddev, 2), 0],
                                [0, 0, 0, 0, 0, 0, np.power(alfadev, 2)]
                                ])  # process uncertainty

        self.my_filter.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                ])  # Measurement function

        self.my_filter.R = np.array([[np.power(xdev_measure, 2), 0, 0, 0, 0],
                                [0, np.power(ydev_measure, 2), 0, 0, 0],
                                [0, 0, np.power(sdev_measure, 2), 0, 0],
                                [0, 0, 0, np.power(ddev_measure, 2), 0],
                                [0, 0, 0, 0, np.power(alfadev_measure, 2)],
                                ])  # state uncertainty

        self.my_filter.P = np.array([[np.power(xdev_init, 2), 0, 0, 0, 0, 0, 0],
                            [0,	0, 0, 0, 0,	0, 0],
                            [0, 0, np.power(ydev_init, 2), 0, 0, 0, 0],
                            [0,	0, 0, 0, 0,	0, 0],
                            [0,	0, 0, 0, pow(sdev_init, 2),	0, 0],
                            [0,	0, 0, 0, 0, pow(ddev_init, 2), 0],
                            [0,	0, 0, 0, 0, 0, pow(alfadev_init, 2)]
                            ])  # coovariance initizialitation

        self.my_filter.x = x_init # initial state (location(x,y), velocity(x,y), acceleration(x,y), and size)

    def predict(self):
        """Predict the future states of the Kalman Filter

        Returns
        -------
        x : np.array
            States of the center of the eyes after prediction
            [x, vx, y, vy, size, distance, rads]

        """

        self.my_filter.predict()

        return self.my_filter.x

    def update(self, measurement):
        """Update the states of the Kalman Filter with a measurement

        Parameters
        -------
        measurement : np.array
            Measurement of the center of the eyes, distance between eyes, and angle of eyes vector
            [x, y, size, distance, rads]

        Returns
        -------
        x : np.array
            States of the center of the eyes after measurement
            [x, vx, y, vy, size, distance, rads]

        """

        self.my_filter.update(measurement)

        return self.my_filter.x


def getCenterOfEyesState(left_eye_state, right_eye_state):
    """
    Returns the states of the center of the eyes.

    Returns the states of the center of the eyes, based on the states of each eye.

    Parameters
    ----------
    left_eye_state : np.array
        Initial states of the left eye:
        [x, vx, ax, y, vy, ay, size]
    right_eye_state : np.array
        Initial states of the right eye:
        [x, vx, ax, y, vy, ay, size]

    Returns
    -------
    np.array
        States of the center of the eyes:
        [x, vx, y, vy, sizes, distance, angle]

    """
    x1 = left_eye_state[0]
    x2 = right_eye_state[0]
    y1 = left_eye_state[3]
    y2 = right_eye_state[3]

    velx1 = left_eye_state[1]
    velx2 = right_eye_state[1]
    vely1 = left_eye_state[4]
    vely2 = right_eye_state[4]

    center = (int((x1+x2)/2), int((y1+y2)/2))
    vel = (int((velx1+velx2)/2), int((vely1+vely2)/2))
    size_pupils = (left_eye_state[6] + right_eye_state[6]) / 2
    dist_between_eyes = math.hypot(x2 - x1, y2 - y1)
    rads_eyes_vector = math.atan2(y2-y1, x2-x1)

    center_of_eyes_state = np.array([center[0], vel[0], center[1], vel[1], size_pupils, dist_between_eyes, rads_eyes_vector])

    return center_of_eyes_state
