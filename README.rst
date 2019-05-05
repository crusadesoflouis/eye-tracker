.. _readme:

****************
Getting Started
****************

This is a Python module for Eyes and Pupils tracking. It can be installed by cloning
the repository and then within the directory, running the pip install command::

  pip install .

After installation, it can be used with these commands::

  eye-tracker-track-pupils FILE [-s] [-o] OUTPUT_FILE
  eye-tracker-track-eyes [-f] FILE [-s] [-o] OUTPUT_FILE

The first one is for tracking pupils with IR video files.
The second one is for tracking eyes with every type of video file, or with the test_webcam
if no file is provided.
The -s option is for showing the debug window, and the -o option is for saving the debug window
to a file.

You can also import this module as a library in your module. For this, please read the Library Usage segment.

If you prefer, you can test the module before installing it, using tox. For this, please read the Testing segment.
