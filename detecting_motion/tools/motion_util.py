"""This module provides utility functions for detecting motion and tracking objects."""

import cv2
import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import dilation, square


class BasicKalmanFilter:
    """Simple Kalman filter implementation for tracking objects.

    This class provides methods for predicting and updating the state of a tracked object.
    """

    def __init__(self) -> None:
        """Initialize the BasicKalmanFilter with default parameters."""
        self.dt = 1  # Time step
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # State transition matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
        self.Q = np.eye(4) * 0.05  # Process noise covariance
        self.R = np.eye(2) * 50000  # Measurement noise covariance
        self.P = np.eye(4) * 200000  # Estimate error covariance
        self.x = np.zeros((4, 1))  # Initial state vector

    def predict(self) -> np.ndarray:
        """Predict the next state of the object.

        Updates the state estimate and the error covariance matrix.

        Returns:
            np.ndarray: The predicted state estimate.
        """
        # Predict state
        self.x = np.dot(self.A, self.x)
        # Predict error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, y: np.ndarray) -> None:
        """Update the Kalman Filter with a new observation.

        Args:
            y (np.ndarray): The new observation vector.
        """
        # Compute residual
        y = y - np.dot(self.H, self.x)
        # Compute residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Compute Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update state estimate
        self.x += np.dot(K, y)
        # Update error covariance
        I = np.eye(self.H.shape[1])
        self.P = np.dot(
            np.dot(I - np.dot(K, self.H), self.P),
            (I - np.dot(K, self.H)).T,
        ) + np.dot(np.dot(K, self.R), K.T)


class ObjectTracker:
    """Tracks a single object using a Kalman filter.

    This class uses a Kalman filter to estimate the state of an object and maintains a history of its positions.
    """

    def __init__(self, kalman_filter: BasicKalmanFilter, inactive_frames: int) -> None:
        """Initialize the ObjectTracker with a Kalman filter and inactivity counter.

        Args:
            kalman_filter (BasicKalmanFilter): The Kalman filter used for tracking.
            inactive_frames (int): The count of frames the object has been inactive.
        """
        self.history = []  # List to store historical positions

        self.filter = kalman_filter
        self.inactive_frames = inactive_frames


class MotionTracker:
    """Detects motion and tracks objects in video frames.

    This class handles the detection of motion in video frames, identifies objects, and maintains their tracking.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self, hysteresis: int, threshold: int, dist_threshold: int, skip_frames: int, max_trackers: int, min_size: int
    ) -> None:
        """Initialize the MotionTracker with specified parameters.

        Args:
            hysteresis (int): Number of frames to keep track of motion.
            threshold (int): Motion detection threshold.
            dist_threshold (int): Distance threshold for object matching.
            skip_frames (int): Number of frames to skip between detections.
            max_trackers (int): Maximum number of objects to track.
            min_size (int): Minimum size of objects to be detected.
        """
        self.hysteresis = hysteresis
        self.threshold = threshold
        self.dist_threshold = dist_threshold
        self.skip_frames = skip_frames
        self.max_trackers = max_trackers
        self.min_size = min_size

        self.frame_buffer = []  # Buffer to store recent frames
        self.trackers = []  # List of tracked objects

    def calculate_motion(self) -> np.ndarray:
        """Calculate the motion frame by comparing consecutive frames.

        Returns:
            np.ndarray: The motion frame after applying the threshold.
        """
        frame1, frame2, frame3 = self.frame_buffer
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)
        motion_frame = cv2.bitwise_and(diff1, diff2)
        _, thresholded_frame = cv2.threshold(motion_frame, self.threshold, 255, cv2.THRESH_BINARY)
        return thresholded_frame

    def find_objects(self, motion_frame: np.ndarray) -> list[regionprops]:
        """Detect objects in the motion frame.

        Args:
            motion_frame (np.ndarray): The frame where motion is detected.

        Returns:
            list[regionprops]: List of detected objects with their properties.
        """
        if len(motion_frame.shape) == 3:
            motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)

        dilated_frame = dilation(motion_frame, square(3))
        labeled_frame = label(dilated_frame)
        detected_objects = regionprops(labeled_frame)
        return [obj for obj in detected_objects if obj.area >= self.min_size]

    def refresh_trackers(self, detected_objects: list[regionprops]) -> None:
        """Update or add new trackers based on detected objects.

        Args:
            detected_objects (list[regionprops]): List of detected objects.
        """
        for obj in detected_objects:
            centroid = np.array([obj.centroid[1], obj.centroid[0], 0, 0]).reshape((4, 1))
            distances = [
                np.linalg.norm(tracker.filter.x[:2] - centroid[:2]) for tracker in self.trackers
            ]

            if distances and min(distances) < self.dist_threshold:
                index = distances.index(min(distances))
                self.trackers[index].filter.update(centroid[:2])
                self.trackers[index].inactive_frames = 0
            elif len(self.trackers) < self.max_trackers:
                kf = BasicKalmanFilter()
                kf.x = centroid
                self.trackers.append(ObjectTracker(kf, 0))

        for tracker in self.trackers:
            tracker.history.append((tracker.filter.x[0, 0], tracker.filter.x[1, 0]))
            tracker.history = tracker.history[-10:]

        self.trackers = [
            tracker for tracker in self.trackers if tracker.inactive_frames <= self.hysteresis
        ]
        for tracker in self.trackers:
            tracker.inactive_frames += 1
            tracker.filter.predict()

    def process_frame(self, frame: np.ndarray, initialize: bool = False) -> None:
        """Process a new frame, updating the frame buffer and tracking objects.

        Args:
            frame (np.ndarray): The new frame to process.
            initialize (bool): Flag indicating if this is for initialization. (Default: False)
        """
        if len(self.frame_buffer) >= 3:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)

        if len(self.frame_buffer) == 3:
            motion_frame = self.calculate_motion()
            if not initialize:
                detected_objects = self.find_objects(motion_frame)
                self.refresh_trackers(detected_objects)

    def process_with_skips(self, frame: np.ndarray, skips: int) -> None:
        """Process a frame with a specified number of frame skips.

        Args:
            frame (np.ndarray): The new frame to process.
            skips (int): Number of frames to skip.
        """
        if skips > 1:
            for _ in range(skips):
                for tracker in self.trackers:
                    tracker.filter.predict()
                    tracker.history.append((tracker.filter.x[0, 0], tracker.filter.x[1, 0]))
                    tracker.history = tracker.history[-10:]
        self.process_frame(frame)

    def initialize(self, frames: list[np.ndarray]) -> None:
        """Initialize the tracker with a set of initial frames.

        Args:
            frames (list[np.ndarray]): List of frames for initialization.

        Raises:
            ValueError: If fewer than 3 frames are provided.
        """
        if len(frames) < 3:
            raise ValueError("At least 3 frames are required for initialization.")
        for frame in frames[:3]:
            self.process_frame(frame, initialize=True)
