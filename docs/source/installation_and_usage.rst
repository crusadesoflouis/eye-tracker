.. installation_and_usage:


************************
Installation and Usage
************************

.. _installation:

Installation
=============================

If you want to install the module in your own virtualenv, you can do it using pip::

  pip install .


.. _usage:

Usage
=============================


This module will create two console scripts in your environment::

  eye-tracker-track-pupils FILE [-s] [-o] OUTPUT_FILE

and::

  eye-tracker-track-eyes [-f] FILE [-s] [-o] OUTPUT_FILE


The first one is for tracking pupils with IR video files.
The second one is for tracking eyes with every type of video file, or with the webcam
if no file is provided.
The -s option is for showing the debug window, and the -o option is for saving the debug window
to a video file.
