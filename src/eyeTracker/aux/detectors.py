# Builtin dependencies
import os

# PyPI dependencies
import cv2
import numpy as np


def getDetectors(dir_classifiers):
    """loads all of cv2 tools"""
    face_detector = cv2.CascadeClassifier(
        os.path.join(dir_classifiers, "haarcascade_frontalface_default.xml"))
    eye_detector = cv2.CascadeClassifier(
        os.path.join(dir_classifiers, 'haarcascade_eye.xml'))
    detector_params = cv2.SimpleBlobDetector_Params()
    # detector_params.minThreshold = params['min_threshold']
    # detector_params.maxThreshold = params['max_threshold']
    # detector_params.thresholdStep = params['max_threshold']
    detector_params.filterByArea = False
    # detector_params.minArea = 100
    # detector_params.maxArea = 500
    detector_params.filterByCircularity = True
    detector_params.minCircularity = 0.85
    detector_params.maxCircularity = 1
    detector_params.filterByConvexity = False
    # detector_params.minConvexity = 0.9
    # detector_params.maxConvexity = 1
    detector_params.filterByInertia = True
    detector_params.minInertiaRatio = 0.3
    detector_params.maxInertiaRatio = 1.0
    detector_params.filterByColor = True
    detector_params.blobColor = 255
    # detector_params.minDistBetweenBlobs
    detector = cv2.SimpleBlobDetector_create(detector_params)

    return face_detector, eye_detector, detector


def detect_face(gray_image, cascade):
    """
    Detects all faces, if multiple found, works with the biggest. Returns the following parameters:
    1. The face frame
    2. A gray version of the face frame
    2. Estimated left eye coordinates range
    3. Estimated right eye coordinates range
    5. X of the face frame
    6. Y of the face frame
    """
    face_rois = cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    gray_face_frame = None
    estimated_left_eye_frame = None
    estimated_right_eye_frame = None
    face_roi = None
    left_eye_estimated_roi = None
    right_eye_estimated_roi = None

    if len(face_rois) >= 1:

        if len(face_rois) > 1:
            face_roi = (0, 0, 0, 0)
            for i in face_rois:
                if i[3] > face_roi[3]: # I keep the widest one
                    face_roi = i
            face_roi = np.array([i], np.int32)
        else:
            face_roi = face_rois

        for (x, y, w, h) in face_roi:
            gray_face_frame = gray_image[y:y + h, x:x + w]

            eye_size_with_margin = (int(w*0.3), int(h/4))
            left_eye_p1 = (x + int(w*0.15), y + int(h/4))
            right_eye_p1 = (x + int(w*0.55), y + int(h/4))

            left_eye_estimated_roi = left_eye_p1 + eye_size_with_margin
            right_eye_estimated_roi = right_eye_p1 + eye_size_with_margin

            estimated_left_eye_frame = gray_face_frame = gray_image[left_eye_p1[1]:left_eye_p1[1] + eye_size_with_margin[1], left_eye_p1[0]:left_eye_p1[0] + eye_size_with_margin[0]]
            estimated_right_eye_frame = gray_face_frame = gray_image[right_eye_p1[1]:right_eye_p1[1] + eye_size_with_margin[1], right_eye_p1[0]:right_eye_p1[0] + eye_size_with_margin[0]]

        face_roi = face_roi[0]

    return gray_face_frame, face_roi, estimated_left_eye_frame, estimated_right_eye_frame, left_eye_estimated_roi, right_eye_estimated_roi


def detect_eye(estimated_eye_frame, cascade):
    """

    :param estimated_eye_frame: estimated eye frame image, based on position inside face frame
    :param cascade: Hhaar cascade for eye
    :return: detected_eye_frame, detected_eye_roi
    """
    detected_eye_frame = None
    detected_eye_roi = None

    eye_rois = cascade.detectMultiScale(estimated_eye_frame, scaleFactor=1.1, minNeighbors=2)

    if len(eye_rois) > 0:

        if len(eye_rois) > 1:
            eye_roi = (0, 0, 0, 0)
            for i in eye_rois:
                if i[3] > eye_roi[3]: # I keep the widest one
                    eye_roi = i
            eye_roi = np.array([i], np.int32)
        else:
            eye_roi = eye_rois

        for (x, y, w, h) in eye_roi:
            reduce_per = 0.3
            y = y + int(h*reduce_per/2)
            h = h - int(h*reduce_per)
            detected_eye_frame = estimated_eye_frame[y:y+h, x:x + w]
            detected_eye_roi = (x, y, w, h)

    return detected_eye_frame, detected_eye_roi


def find_pupil_keypoints(eye_frame, detector, prevArea=None, webcam=False):
    """

    :param eye_frame: eye frame image, detected by Hhaar cascade of eye
    :param cascade: Hhaar cascade
    :return: detected_eye_frame, detected_eye_roi
    """

    if webcam:
        _, processed_eye_frame = cv2.threshold(eye_frame, 100, 255, cv2.THRESH_BINARY)
    else:
        _, processed_eye_frame = cv2.threshold(eye_frame, 230, 255, cv2.THRESH_BINARY)
    processed_eye_frame = cv2.dilate(processed_eye_frame, None, iterations=4)
    processed_eye_frame = cv2.erode(processed_eye_frame, None, iterations=2)
    processed_eye_frame = cv2.medianBlur(processed_eye_frame, 5)
    keypoints = detector.detect(processed_eye_frame)

    if keypoints and prevArea and len(keypoints) > 1:
        tmp = 1000
        for keypoint in keypoints:  # filter out odd blobs
            if abs(keypoint.size - prevArea) < tmp:
                ans = keypoint
                tmp = abs(keypoint.size - prevArea)
        keypoints = np.array([ans])

    return keypoints, processed_eye_frame


def detect_pupil(eye_frame, detector, previous_area=None, webcam=False):
    """
    Detects pupil from in an eye frame.


    Parameters
    ----------
    eye_frame : np.array
        OpenCV image, with the frame of one eye.
    detector: cv2.SimpleBlobDetector
        Blob Detector for the pupils
    previous_area : int
        Previous area of the eye. It is used for better accuracy of the SimpleBlobDetector.
    webcam : boolean
        If it's from webcam, I change some detection parameters.

    Returns
    -------
    pupil_keypoints: OpenCV keypoints
        Keypoints of the current frame
    pupil_position: np.array
        Current position of the eye.
    pupil_area: int
        Current area of the eye.
    processed_eye_frame: np.array
        OpenCV image, with the processed frame of one eye.
    """
    pupil_keypoints, processed_eye_frame = find_pupil_keypoints(eye_frame, detector,
                                    prevArea=previous_area, webcam=webcam)

    if pupil_keypoints:
        pupil_position = None
        pupil_area = 0
        for keypoint in pupil_keypoints:
            if keypoint.size > pupil_area:
                pupil_position = keypoint.pt
                pupil_area = keypoint.size
    else:
        pupil_position = None
        pupil_area = None

    return pupil_keypoints, pupil_position, pupil_area, processed_eye_frame
