from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Builtin dependencies
import operator
import math
import logging
import os

# PyPI dependencies
import cv2
import numpy as np
import imutils

# Local dependencies
from eyeTracker.aux import detectors
from eyeTracker.aux.bothEyesTracker import BothEyesTracker
from eyeTracker.aux.oneEyeTracker import OneEyeTracker
from eyeTracker.aux.bothEyesTracker import getCenterOfEyesState


# Main class
class eyeTracker(object):
    """Eye Tracker class

    An object of this class will be instanced by a script, and then the trackPupils will be called.

    Parameters
    ----------
    left_eye_init_state : np.array
        Initial states of the left eye:
        [x, vx, ax, y, vy, ay, size]
    right_eye_init_state : np.array
        Initial states of the right eye:
        [x, vx, ax, y, vy, ay, size]

    """

    # CONSTRUCTOR
    def __init__(self, left_eye_init_state=None, right_eye_init_state=None):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_classifiers = os.path.join(dir_path, "data/Classifiers/haar")

        self.face_detector, self.eye_detector, self.pupil_detector = detectors.getDetectors(dir_classifiers=dir_classifiers)
        self.camera_is_running = False
        self.previous_keypoints = [None, None]
        self.previous_pupil_area = [None, None]
        self.previous_pupil_position = [None, None]
        self.previous_mode = [None, None]
        if left_eye_init_state==None and right_eye_init_state==None:
            # Set default values
            left_eye_init_state = np.array([180, 0, 0, 320, 0, 0, 5])
            right_eye_init_state = np.array([300, 0, 0, 320, 0, 0, 5])
        # self.previous_pupil_area = [left_eye_init_state[6], left_eye_init_state[6]]
        # self.previous_pupil_position = [(left_eye_init_state[0], left_eye_init_state[3]), (right_eye_init_state[0], right_eye_init_state[3])]

        center_of_eyes_init_state = getCenterOfEyesState(left_eye_init_state, right_eye_init_state)

        self.eye_trackers = [OneEyeTracker(left_eye_init_state),
                             OneEyeTracker(right_eye_init_state)]
        self.center_of_eyes_tracker = BothEyesTracker(center_of_eyes_init_state)


    def startTracking(self, track_pupils=False, source=None, show=True, output=None):
        """
        Track the eyes/pupils.

        Starts the capture, and makes the loop.

        Parameters
        ----------
        pupils_or_eyes : boolean
            Whether to track pupils or just eyes
        source : string
            string: path of video file
        show : boolean
            Shows the debug window
        output : None / string
            Path of output file

        """

        if track_pupils:
            if source:
                capture = cv2.VideoCapture(source)
                if not capture:
                    logging.error("Error opening file, or file not found.")
            else:
                logging.error("A file source with an IR video must be provided for pupils tracking")
        elif not source:
                capture = cv2.VideoCapture(0)
                if not capture:
                    logging.error("No camera found.")
        else:
            capture = cv2.VideoCapture(source)
            if not capture:
                logging.error("Error opening file, or file not found.")

        if capture.isOpened():
            return_ok, base_image = capture.read()
            if return_ok:

                if source is not None:
                    base_image = imutils.rotate_bound(base_image, -90)
                else:
                    webcam = True

                if output:
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps=30.0, frameSize=(base_image.shape[1],base_image.shape[0]), isColor=True)

                # Loop
                while(return_ok and capture.isOpened()):
                    if track_pupils:
                        debug_image = self.processFrameForPupilsTracking(frame=base_image,
                                                        face_detector=self.face_detector,
                                                        eye_detector=self.eye_detector,
                                                        pupil_detector=self.pupil_detector,
                                                        pupil_trackers=self.eye_trackers,
                                                        center_of_pupils_tracker=self.center_of_eyes_tracker)
                    else:
                        debug_image, _, _, _ = self.processFrameForEyesTracking(frame=base_image,
                                                        face_detector=self.face_detector,
                                                        eye_detector=self.eye_detector,
                                                        eye_trackers=self.eye_trackers,
                                                        center_of_eyes_tracker=self.center_of_eyes_tracker,
                                                        webcam=webcam)

                    if (show):
                        cv2.imshow('Debug', debug_image)
                    if (output):
                        writer.write(debug_image)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    return_ok, base_image = capture.read()
                    if return_ok is True and source is not None:
                        base_image = imutils.rotate_bound(base_image, -90)

                logging.info('Video capture ended.')
                capture.release()
                if output:
                    writer.release()
                cv2.destroyAllWindows()
            else:
                logging.error('Error capturing video.')
        else:
            logging.error('Error opening video capture.')
        return


    def processFrameForPupilsTracking(self, frame, face_detector, eye_detector, pupil_detector, pupil_trackers, center_of_pupils_tracker):
        """
        Processes the image for tracking the pupils of eyes.

        This is the core function for pupils tracking.
        It first calls processFrameForEyesTracking, for tracking the eyes, and then it uses a SimpleBlobDetector to look for the pupils (blue circle)
        If given, two Kalman filters track each pupil state independently (green circle), and another Kalman filter (red circle) tracks the state of
        both pupils (it has another model) based on information of the output of the other two trackers.

        Parameters
        ----------
        frame : np.array
            OpenCV image
        face_detector : cv2.CascadeClassifier
            Hhaar cascade detector for the face
        eye_detector : cv2.CascadeClassifier
            Hhaar cascade detector for the eyes
        pupil_detector : cv2.SimepleBlobDetector
            Blob detector for the pupils
        pupil_trackers: [KalmanFilter, KalmanFilter]
            KalmanFilters for the pupils
        center_of_pupils_tracker: KalmanFilter
            KalmanFilter for the center of the pupils
        webcam : boolean
            If it's from webcam, I change some detection parameters.

        Returns
        -------
        np.array
            debug_image
            Image with drawings for debuging

        """

        debug_image, detected_eye_frames, estimated_eye_rois, detected_eye_rois = self.processFrameForEyesTracking(frame=frame,
                                                                            face_detector=face_detector,
                                                                            eye_detector=eye_detector,
                                                                            eye_trackers=None,
                                                                            center_of_eyes_tracker=None,
                                                                            webcam=False)

        measured_pupil_position = [None, None]
        measured_pupil_size = [None, None]
        pupil_states = [None, None]
        tracked_pupil_position = [None, None]
        tracked_pupil_vel = [None, None]
        tracked_pupil_size = [None, None]

        for i in range(2):

            if (detected_eye_frames[i]) is not None:

                pupil_keypoints, pupil_position, pupil_area, _ = detectors.detect_pupil(
                    eye_frame=detected_eye_frames[i],
                    detector=pupil_detector,
                    previous_area=self.previous_pupil_area[i])

                if not pupil_keypoints:
                    pupil_keypoints = self.previous_keypoints[i]
                    pupil_position = self.previous_pupil_position[i]
                    pupil_area = self.previous_pupil_area[i]

                if pupil_position is not None:
                    int_tuple_position = (int(pupil_position[0]), int(pupil_position[1]))
                    measured_pupil_position[i] = tuple(map(operator.add, estimated_eye_rois[i][:2], int_tuple_position))
                    measured_pupil_position[i] = tuple(map(operator.add, detected_eye_rois[i][:2], measured_pupil_position[i]))
                    if pupil_area is not None:
                        measured_pupil_size[i] = pupil_area

                self.previous_keypoints[i] = pupil_keypoints
                self.previous_pupil_position[i] = pupil_position
                self.previous_pupil_area[i] = pupil_area

            if pupil_trackers is not None:

                pupil_states[i] = pupil_trackers[i].predict()

                if (measured_pupil_size[i]) is not None:
                    z = np.array([measured_pupil_position[i][0], measured_pupil_position[i][1], int(measured_pupil_size[i])])
                    cv2.circle(debug_image, center=measured_pupil_position[i], radius=int(measured_pupil_size[i]), color=(255,0,0), thickness=1, lineType=8, shift=0)
                    pupil_states[i] = pupil_trackers[i].update(z)

                tracked_pupil_position[i] = (int(pupil_states[i][0]), int(pupil_states[i][3]))
                tracked_pupil_vel[i] = (int(pupil_states[i][1]), int(pupil_states[i][4]))
                tracked_pupil_size[i] = int(pupil_states[i][6])

                cv2.circle(debug_image, center=tracked_pupil_position[i], radius=tracked_pupil_size[i], color=(0,255,0), thickness=1, lineType=8, shift=0)
                cv2.arrowedLine(debug_image, tracked_pupil_position[i], tuple(map(operator.add, tracked_pupil_position[i], tracked_pupil_vel[i])), color=(0,255,0), thickness=1, shift=0	)

        if pupil_trackers is not None and center_of_pupils_tracker is not None:

            tracked_center_of_eyes = center_of_pupils_tracker.predict()

            measured_center_of_eyes = getCenterOfEyesState(pupil_states[0], pupil_states[1])
            z = np.array([measured_center_of_eyes[0], measured_center_of_eyes[2], measured_center_of_eyes[4], measured_center_of_eyes[5], measured_center_of_eyes[6]])
            tracked_center_of_eyes = center_of_pupils_tracker.update(z)

            center = (int(tracked_center_of_eyes[0]), int(tracked_center_of_eyes[2]))
            vel = (int(tracked_center_of_eyes[1]), int(tracked_center_of_eyes[3]))
            size = tracked_center_of_eyes[4]
            dist = tracked_center_of_eyes[5]
            rads = tracked_center_of_eyes[6]

            x2 = int(center[0] + math.cos(rads)*dist/float(2))
            x1 = int(center[0] - math.cos(rads)*dist/float(2))
            y2 = int(center[1] + math.sin(rads)*dist/float(2))
            y1 = int(center[1] - math.sin(rads)*dist/float(2))

            cv2.circle(debug_image, center=(x1, y1), radius=int(size), color=(0,0,255), thickness=1, lineType=8, shift=0)
            cv2.circle(debug_image, center=(x2, y2), radius=int(size), color=(0,0,255), thickness=1, lineType=8, shift=0)
            cv2.circle(debug_image, center=center, radius=int(1), color=(0,0,255), thickness=2, lineType=8, shift=0)
            cv2.line(debug_image, (x1, y1), (x2, y2), color=(0,0,255), thickness=1, lineType=8, shift=0)
            cv2.arrowedLine(debug_image, center, tuple(map(operator.add, center,vel)), color=(0,0,255), thickness=1, shift=0)

        return debug_image



    def processFrameForEyesTracking(self, frame, face_detector, eye_detector, eye_trackers=None, center_of_eyes_tracker=None, webcam=True):
        """
        Processes the image for tracking the eyes.

        This is the core function for eyes tracking.
        It implements a Haar Classifier for the face (red ROI), then it estimates a region for possible possition of the eyes (green ROIs),
        then it looks for each eyes within those regions with two Haar Classifiers (blue ROIs). If given, two Kalman filters track each eye state
        independently (green arrows), and another Kalman filter (red arrows) tracks the state of both eyes (it has another model) based on information
        of the output of the other two trackers.

        Parameters
        ----------
        frame : np.array
            OpenCV image
        face_detector : cv2.CascadeClassifier
            Hhaar cascade detector for the face
        eye_detector : cv2.CascadeClassifier
            Hhaar cascade detector for the eyes
        eye_trackers: [KalmanFilter, KalmanFilter]
            KalmanFilters for the eyes
        center_of_eyes_tracker: KalmanFilter
            KalmanFilter for the center of the eyes
        webcam : boolean
            If it's from webcam, I change some detection parameters.

        Returns
        -------
        debug_image: np.array
            Image with drawings for debuging
        detected_eye_frames: [np.array, np.array]
            Liet with two OpenCV images, one for each eye frame

        """

        detected_eye_frames = [None, None]
        estimated_eye_frames = [None, None]
        detected_eye_rois = [None, None]
        estimated_eye_rois = [None, None]
        tracked_eye_position = [None, None]
        tracked_eye_vel = [None, None]
        tracked_eye_size = [None, None]

        debug_image = frame.copy()
        gray_image = frame.copy()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        gray_face_frame, face_roi, estimated_eye_frames[0], estimated_eye_frames[1], estimated_eye_rois[0], estimated_eye_rois[1] = detectors.detect_face(
            gray_image, face_detector)

        if face_roi is not None:
            x = face_roi[0]
            y = face_roi[1]
            w = face_roi[2]
            h = face_roi[3]
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0,0,255), thickness=1, lineType=8, shift=0)

        eye_states = [None, None]

        for i in range(2):

            if estimated_eye_frames[i] is not None:

                eye_p1 = estimated_eye_rois[i][:2]
                eye_p2 = tuple(map(operator.add, eye_p1, estimated_eye_rois[0][2:]))
                cv2.rectangle(debug_image, eye_p1, eye_p2, (0,255,0), thickness=1, lineType=8, shift=0)

                detected_eye_frames[i], detected_eye_rois[i] = detectors.detect_eye(estimated_eye_frames[i], eye_detector)

                if detected_eye_frames[i] is not None:

                    eye_p1 = tuple(map(operator.add, detected_eye_rois[i][:2], estimated_eye_rois[i][:2]))
                    eye_p2 = tuple(map(operator.add, eye_p1, detected_eye_rois[i][2:]))
                    eye_center = tuple(map(operator.add, eye_p1, tuple(val/2 for val in detected_eye_rois[i][2:])))
                    cv2.rectangle(debug_image, eye_p1, eye_p2, (255,0,0), thickness=1, lineType=8, shift=0)

            if eye_trackers is not None:

                eye_states[i] = eye_trackers[i].predict()

                if detected_eye_frames[i] is not None:
                    z = np.array([eye_center[0], eye_center[1], 0])
                    eye_states[i] = eye_trackers[i].update(z)

                tracked_eye_position[i] = (int(eye_states[i][0]), int(eye_states[i][3]))
                tracked_eye_vel[i] = (int(eye_states[i][1]), int(eye_states[i][4]))
                tracked_eye_size[i] = int(eye_states[i][6])

                cv2.arrowedLine(debug_image, tracked_eye_position[i], tuple(map(operator.add, tracked_eye_position[i], tracked_eye_vel[i])), color=(0,255,0), thickness=1, shift=0	)

        if eye_trackers is not None and center_of_eyes_tracker is not None:

            tracked_center_of_eyes = center_of_eyes_tracker.predict()

            measured_center_of_eyes = getCenterOfEyesState(eye_states[0], eye_states[1])
            z = np.array([measured_center_of_eyes[0], measured_center_of_eyes[2], measured_center_of_eyes[4], measured_center_of_eyes[5], measured_center_of_eyes[6]])
            tracked_center_of_eyes = center_of_eyes_tracker.update(z)

            center = (int(tracked_center_of_eyes[0]), int(tracked_center_of_eyes[2]))
            vel = (int(tracked_center_of_eyes[1]), int(tracked_center_of_eyes[3]))
            # size = tracked_center_of_eyes[4]
            dist = tracked_center_of_eyes[5]
            rads = tracked_center_of_eyes[6]

            x2 = int(center[0] + math.cos(rads)*dist/float(2))
            x1 = int(center[0] - math.cos(rads)*dist/float(2))
            y2 = int(center[1] + math.sin(rads)*dist/float(2))
            y1 = int(center[1] - math.sin(rads)*dist/float(2))

            cv2.arrowedLine(debug_image, (x1, y1), tuple(map(operator.add, (x1, y1), vel)), color=(0,0,255), thickness=1, shift=0	)
            cv2.arrowedLine(debug_image, (x2, y2), tuple(map(operator.add, (x2, y2), vel)), color=(0,0,255), thickness=1, shift=0	)
            cv2.circle(debug_image, center=center, radius=int(1), color=(0,0,255), thickness=2, lineType=8, shift=0)
            cv2.line(debug_image, (x1, y1), (x2, y2), color=(0,0,255), thickness=1, lineType=8, shift=0)
            cv2.arrowedLine(debug_image, center, tuple(map(operator.add, center, vel)), color=(0,0,255), thickness=1, shift=0)

        return debug_image, detected_eye_frames, estimated_eye_rois, detected_eye_rois
