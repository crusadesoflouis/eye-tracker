from __future__ import print_function

# Python dependencies:
import os
import logging

# PIP dependencies:
import pbr.version
from nose.plugins.attrib import attr

from eyeTracker.console.track_pupils import _main as track_pupils
from eyeTracker.console.track_eyes import _main as track_eyes

video_1 = 'data/in/OcuIR_Inside_MostlyArtificial_TowardsCamera_home_CameraTop_10s_20190316_092357.mp4'
video_2 = 'data/in/OcuIR_Inside_MostlyArtificial_TowardsCamera_office_CameraTop_10s_20190318_095000.mp4'

FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
logging.basicConfig(format=FORMAT, level = logging.INFO)

class testModule(object):

    # Directorio a lee/escribir para test
    dir_path = os.path.dirname(os.path.realpath(__file__))
    version = pbr.version.VersionInfo("module").version_string()

    def setUp(self):
        self.args = {}
        self.args['show'] = True
        pass

    def tearDown(self):
        pass

    def aux_function(self):
        return

    @attr(attr='1')
    @attr('video_1')
    def test_1(self):
        self.args['file'] = os.path.join(self.dir_path, video_1)
        self.args['output'] = os.path.join(self.dir_path, 'data/out/video_1.mp4')
        track_pupils(self.args)
        pass

    @attr(attr='2')
    @attr('video_2')
    def test_2(self):
        self.args['file'] = os.path.join(self.dir_path, video_2)
        self.args['output'] = os.path.join(self.dir_path, 'data/out/video_2.mp4')
        track_pupils(self.args)
        pass

    @attr(attr='3')
    @attr('webcam')
    def test_webcam(self):
        self.args['file'] = None
        self.args['output'] = os.path.join(self.dir_path, 'data/out/webcam.mp4')
        track_eyes(self.args)
        pass
