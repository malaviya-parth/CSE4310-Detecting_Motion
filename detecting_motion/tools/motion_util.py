"""This module provides the utility Functions for motion detection and object tracking."""

import cv2
import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import dilation, square


class KalmanFilter:
    """A basic Kalman filter implementation for object tracking.

    This class provides methods for predicting and updating the state of a tracked object.
    """

    def __init__(self) -> None:
        """Initialize the Kalman Filter with default parameters.

        The parameters include time step, state transition matrix, observation matrix,
        process noise covariance, measurement noise covariance, estimate error covariance,
        and initial state.
        """
        self.dt = 1  # Time step
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # Transition matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
        self.Q = np.eye(4) * 0.05  # Process noise covariance
        self.R = np.eye(2) * 50000  # Measurement noise covariance
        self.P = np.eye(4) * 200000  # Estimate error covariance
        self.x = np.zeros((4, 1))  # Initial state vector

    def predict(self) -> np.ndarray:
        """Perform the prediction step of the Kalman Filter.

        Updates the state estimate and estimate error covariance using the state transition matrix.

        Returns:
            np.ndarray: Updated state estimate after the prediction step.
        """
        # State Prediction
        self.x = np.dot(self.A, self.x)
        # Estimate Error Covariance Prediction
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, y: np.ndarray) -> None:
        """Perform the update step of the Kalman Filter.

        Incorporates the new observation into the state estimate.

        Args:
            y (np.ndarray): The new observation used for updating the filter.
        """
        # Observation Residual
        y = y - np.dot(self.H, self.x)
        # Residual Covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # Update State Estimate
        self.x += np.dot(K, y)
        # Update Estimate Error Covariance
        identity_matrix = np.eye(self.H.shape[1])
        self.P = np.dot(
            np.dot(identity_matrix - np.dot(K, self.H), self.P),
            (identity_matrix - np.dot(K, self.H)).T,
        ) + np.dot(np.dot(K, self.R), K.T)


class TrackedObject:
    """Represents a single tracked object in the motion detection system.

    This class keeps track of an object's state using a Kalman filter, and stores its
    history of positions and the number of frames it has been inactive.
    """

    def __init__(self, kalman_filter: KalmanFilter, inactive_frames: int) -> None:
        """Initialize a TrackedObject with a Kalman filter and an inactive frame counter.

        Args:
            kalman_filter (KalmanFilter): The Kalman filter associated with the tracked object.
            inactive_frames (int): The number of frames for which the object has been inactive.
        """
        self.filter = kalman_filter
        self.inactive_frames = inactive_frames
        # Initialize an empty list to store past positions
        self.previous_positions = []


class MotionDetection:
    """Motion detection and object tracking system.

    This class handles motion detection in video frames, object detection, and maintaining
    a list of tracked objects with the help of a Kalman filter.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self, A: int, T: int, D: int, S: int, N: int, size_threshold: int
    ) -> None:
        """Initialize the MotionDetection class with specified hyperparameters.

        Args:
            A (int): Frame hysteresis.
            T (int): Motion threshold.
            D (int): Distance threshold.
            S (int): Number of frames to skip between detections.
            N (int): Maximum number of objects.
            size_threshold (int): Size threshold for object detection.
        """
        # Hyperparameters
        self.frame_hysteresis = A
        self.motion_threshold = T
        self.distance_threshold = D
        self.frame_skip = S
        self.max_objects = N
        self.size_threshold = size_threshold

        # State variables
        self.frame_buffer = []  # type: list[np.ndarray]
        self.tracked_objects = []  # type: list[TrackedObject]

    def compute_motion(self) -> np.ndarray:
        """Compute the motion frame based on differences between consecutive frames.

        Returns:
            np.ndarray: The motion frame computed from the current frame buffer.
        """
        frame1, frame2, frame3 = self.frame_buffer
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)
        motion_frame = cv2.bitwise_and(diff1, diff2)
        _, threshold = cv2.threshold(motion_frame, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return threshold

    def detect_objects(self, motion_frame: np.ndarray) -> list[regionprops]:
        """Detect objects in the motion frame.

        Args:
            motion_frame (np.ndarray): The motion frame where objects are to be detected.

        Returns:
            list[regionprops]: A list of detected objects.
        """
        # Ensure motion_frame is 2D
        if len(motion_frame.shape) == 3:
            motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)

        # Detect objects in the motion frame
        dilated_frame = dilation(motion_frame, square(3))
        labeled_frame = label(dilated_frame)
        objects = regionprops(labeled_frame)
        return [obj for obj in objects if obj.area >= self.size_threshold]

    def update_tracked_objects(self, detected_objects: list[regionprops]) -> None:
        """Update the list of tracked objects based on detected objects.

        Args:
            detected_objects (list[regionprops]): A list of newly detected objects.
        """
        for obj in detected_objects:
            centroid = np.array([obj.centroid[1], obj.centroid[0], 0, 0]).reshape((4, 1))
            distances = [
                np.linalg.norm(kf.filter.x[:2] - centroid[:2]) for kf in self.tracked_objects
            ]

            if distances and min(distances) < self.distance_threshold:
                index = distances.index(min(distances))
                self.tracked_objects[index].filter.update(centroid[:2])
                self.tracked_objects[index].inactive_frames = 0
            elif len(self.tracked_objects) < self.max_objects:
                kf = KalmanFilter()
                kf.x = centroid
                self.tracked_objects.append(TrackedObject(kf, 0))

        for obj in self.tracked_objects:
            # Update the previous positions list
            obj.previous_positions.append((obj.filter.x[0, 0], obj.filter.x[1, 0]))
            # Keep last 10 positions
            obj.previous_positions = obj.previous_positions[-10:]

        # Remove inactive objects
        self.tracked_objects = [
            obj for obj in self.tracked_objects if obj.inactive_frames <= self.frame_hysteresis
        ]
        for obj in self.tracked_objects:
            obj.inactive_frames += 1
            obj.filter.predict()

    def reset_objects(self) -> None:
        """Reset the object tracking.

        This method clears the current list of tracked objects, effectively reinitializing
        the object tracking state. Used for starting a new tracking session or
        when the scene significantly changes.
        """
        self.tracked_objects = []

    def update(self, frame: np.ndarray, initialize: bool = False) -> None:
        """Update the frame buffer and detect objects if necessary.

        Args:
            frame (np.ndarray): The new frame to be added to the buffer.
            initialize (bool): Flag to indicate if this is part of initialization. (Default: False)
        """
        if len(self.frame_buffer) >= 3:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)

        if len(self.frame_buffer) == 3:
            # Compute the motion frame
            motion_frame = self.compute_motion()
            if not initialize:
                # Detect and update object candidates
                detected_objects = self.detect_objects(motion_frame)
                self.update_tracked_objects(detected_objects)

    def update_with_skips(self, frame: np.ndarray, skips: int) -> None:
        """Update the frame buffer and object tracking with a specified number of frame skips.

        Args:
            frame (np.ndarray): The new frame to be processed.
            skips (int): The number of frames skipped.
        """
        if skips > 1:
            # Run predict step multiple times to simulate time passing
            for _ in range(skips):
                for obj in self.tracked_objects:
                    obj.filter.predict()
                    # Add predicted position to the trail
                    obj.previous_positions.append((obj.filter.x[0, 0], obj.filter.x[1, 0]))
                    obj.previous_positions = obj.previous_positions[-10:]  # Limit trail history
        self.update(frame)

    def initialize(self, frames: list[np.ndarray]) -> None:
        """Initialize the detector with the first set of frames.

        Args:
            frames (list[np.ndarray]): List of initial frames for the detector.

        Raises:
            ValueError: If less than 3 frames are provided for initialization.
        """
        if len(frames) < 3:
            raise ValueError("At least 3 frames are needed for initialization")
        for frame in frames[:3]:
            self.update(frame, initialize=True)
