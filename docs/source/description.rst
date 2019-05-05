.. _description:


************************
Description
************************

.. _implementation:

Implementation
=============================

This modules implements the following sequence to track the eyes/pupils:

1.  It uses Haar Cascade classifiers to look for a face in the video.

2.  If a face is found, then it creates two ROIs to look for the eyes.

3.  En each of these ROIs, one Haar Cascade classifier is used to look for an eye.

4.
    a) If tracking eyes, then two Kalman Filters (one for each eye, and working
    independently) is used to better estimate their states (position and velocity).
    When a eye is detected in one of the ROIs, an update and a prediction is made.
    If no eye is detected in a frame, then the corresponding Kalman Filter just
    makes the prediction (without update, because no measurement was done).

    b) If tracking pupils, then a blob detector is used to look for the pupils.
    Then two Kalman Filters (one for each pupil, and working independently) is
    used to better estimate their states (position, velocity and size). Then the
    sequence for filtering is the same as 4a.

5.  Another Kalman Filter (with a different prediction and measurement model) is
used to track BOTH eyes/pupils. It adds to the states the distance and angle
between the eyes/pupils, so to make better predictions of their positions.
