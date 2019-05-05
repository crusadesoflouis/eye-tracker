.. _library:

****************
Library Usage
****************

If you want to use this module as a library, extending its usage beyond the console scripts,
you can read about the classes and scripts that it includes here.

.. _main_class:

Main Eye Tracker Class
======================

This file contains the main class. An object of this class should be initialized
in a Python script for tracking the eyes. Then, the method startTracking should
be called. An example is given below::

  pupils_tracker = eyeTracker()
  pupils_tracker.startTracking(track_pupils=True, source=FILE, show=True, output=OUTPUT_FILE)
  eyes_tracker = eyeTracker()
  eyes_tracker.startTracking(source=FILE, show=False)

Tha class also has the methods processFrameForPupilsTracking and processFrameForEyesTracking,
which implement the actual computer vision flow.

.. automodule:: eyeTracker.eyeTracker
   :members:

.. _detectors:

Detectors
===================

This file contains the detectors. It implements a Haar Classifiers for the face and the eyes,
and a Blob Detector for the pupils.

.. automodule:: eyeTracker.aux.detectors
  :members:

.. _trackers:

Trackers
===================

This file contains the Kalman filters for tracking. One class is for a Kalman filter for each eye
(i.e: they track the eyes independently from each other), and the other one is a Kalman filter for
tracking both eyes (i.e: it receives as input the ouputs of the other two, but takes into account
the relative position between them, and, in case fo tracking pupils, averages their two sizes).

.. automodule:: eyeTracker.aux.oneEyeTracker
  :members:
.. automodule:: eyeTracker.aux.bothEyesTracker
  :members:
