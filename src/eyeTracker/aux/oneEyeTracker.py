from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# PyPI dependencies
import numpy as np
from filterpy.kalman import KalmanFilter


class OneEyeTracker():
    """One Eye Tracker using a Kalman Filter

    Parameters
    ----------
    x_init : np.array
        Initial states of the the eye:
        [x, vx, ax, y, vy, ay, size]

    """
    def __init__(self, x_init):

    # SETTINGS

        # Prediction Errors
        axdev = 10 # pixels/s^2
        aydev = 10 # pixels/s^2
        sdev = 10 # pixels

        # Measurement Errors
        xdev_measure = 5
        ydev_measure = 5
        sdev_measure = 2

        # Initial Errors of States
        xdev_init = 240
        ydev_init = 320
        sdev_init = 10

        # FPS
        fps = 30

    # KALMAN

        # Initialize the filter's matrices
        self.my_filter = KalmanFilter(dim_x=7, dim_z=3)

        self.my_filter.F = np.array([[1, 1/float(fps), np.power(1/float(fps), 2)/2, 0, 0, 0, 0],
                                [0, 1, 1/float(fps), 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1/float(fps), np.power(1/float(fps), 2)/2, 0],
                                [0, 0, 0, 0, 1, 1/float(fps), 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]
                                ])  # state transition matrix

        self.my_filter.Q = np.array([[np.power(axdev, 2)*np.power(1/float(fps), 4)/36, np.power(axdev, 2)*np.power(1/float(fps),3)/12, np.power(axdev, 2)*np.power(1/float(fps), 2)/6, 0, 0, 0, 0],
                                [np.power(axdev, 2)*np.power(1/float(fps), 3)/12, np.power(axdev, 2)*np.power(1/float(fps), 2)/4, np.power(axdev, 2)*1/float(fps)*1/2, 0, 0, 0, 0],
                                [np.power(axdev, 2)*np.power(1/float(fps), 2)/6, np.power(axdev, 2)*1/float(fps)*1/2, np.power(axdev, 2), 0, 0, 0, 0],
                                [0, 0, 0, np.power(aydev, 2)*np.power(1/float(fps), 4)/36, np.power(aydev, 2)*np.power(1/float(fps),3)/12, np.power(aydev, 2)*np.power(1/float(fps), 2)/6, 0],
                                [0, 0, 0, np.power(aydev, 2)*np.power(1/float(fps), 3)/12, np.power(aydev, 2)*np.power(1/float(fps), 2)/4, np.power(aydev, 2)*1/float(fps)*1/2, 0],
                                [0, 0, 0, np.power(aydev, 2)*np.power(1/float(fps), 2)/6, np.power(aydev, 2)*1/float(fps)*1/2, np.power(aydev, 2), 0],
                                [0, 0, 0, 0, 0, 0, np.power(sdev, 2)]
                                ])  # process uncertainty

        self.my_filter.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1]
                                ])  # Measurement function

        self.my_filter.R = np.array([[np.power(xdev_measure, 2), 0, 0],
                                [0, np.power(ydev_measure, 2), 0],
                                [0, 0, np.power(sdev_measure, 2)]
                                ])  # state uncertainty

        self.my_filter.P = np.array([[np.power(xdev_init, 2),	0, 0, 0, 0,	0, 0],
                            [0,	0, 0, 0, 0,	0, 0],
                            [0,	0, 0, 0, 0,	0, 0],
                            [0, 0, 0, np.power(ydev_init, 2), 0, 0, 0],
                            [0,	0, 0, 0, 0,	0, 0],
                            [0,	0, 0, 0, 0,	0, 0],
                            [0,	0, 0, 0, 0,	0, pow(sdev_init, 2)]
                            ])  # coovariance initizialitation

        self.my_filter.x = x_init # initial state (location(x,y), velocity(x,y), acceleration(x,y), and size)

    def predict(self):
        """Predict the future states of the Kalman Filter

        Returns
        -------
        x : np.array
            States of the of the eye after prediction
            [x, vx, ax, y, vy, ay, size]

        """

        self.my_filter.predict()

        return self.my_filter.x

    def update(self, measurement):
        """Update the states of the Kalman Filter with a measurement

        Parameters
        -------
        measurement : np.array
            Measurement of the states of the eye
            [x, y, size]

        Returns
        -------
        x : np.array
            States of the of the eye after measurement
            [x, vx, ax, y, vy, ay, size]

        """

        self.my_filter.update(measurement)

        return self.my_filter.x
