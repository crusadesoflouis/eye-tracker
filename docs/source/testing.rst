.. _testing:


***************
Testing
***************

.. _tests:

Tests
=============================

There are 3 tests within this module. Two for the IR videos provided (the pupils
will be tracked), and one to test the module with a webcam (the eyes will be
tracked).

.. _test_with_tox:

Testing with tox
=============================

tox is a generic virtualenv management and test command line tool. It creates
automatically a Python virtual environment, and test the module inside
it. So you don't need to create new environment just to test this module.
To test the module with tox, first install tox::

  pip install tox

And then run::

  tox
  
