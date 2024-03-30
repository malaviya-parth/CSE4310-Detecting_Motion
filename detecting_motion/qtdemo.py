import argparse
import sys
import time

from threading import Thread

import cv2
import numpy as np
import skvideo

from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square


# Set the path to the FFmpeg binary, which is required by skvideo
#   -- Set a try and except later so that it can be used on different systems if installed
skvideo.setFFmpegPath(
    "C:\\Users\\Administrator\\Desktop\\School\\3_Detecting_Motion\\ffmpeg_essentials_build\\bin"
)

from skvideo.io import vread


np.float = np.float64
np.int = np.int_


class KalmanFilter:
    """A simple implementation of the Kalman Filter for object tracking.

    This class provides methods for predicting and updating the state of a tracked object
    based on linear motion models.
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
        )  # State transition matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
        self.Q = np.eye(4) * 0.05  # Process noise covariance
        self.R = np.eye(2) * 50000  # Measurement noise covariance
        self.P = np.eye(4) * 200000  # Estimate error covariance
        self.x = np.zeros((4, 1))  # Initial state (x, y, vx, vy)

    def predict(self) -> np.ndarray:
        """Perform the prediction step of the Kalman Filter.

        Updates the state estimate and estimate error covariance using the state transition matrix.

        Returns:
            np.ndarray: Updated state estimate after the prediction step.
        """
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z: np.ndarray) -> None:
        """Perform the update step of the Kalman Filter.

        Incorporates the new observation into the state estimate.

        Args:
            z (np.ndarray): The new observation used for updating the filter.
        """
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(
            np.dot(K, self.R), K.T
        )


class MotionDetection:
    """Motion detection and object tracking system.

    This class handles motion detection in video frames, object detection, and maintaining
    a list of tracked objects with the help of a Kalman filter.
    """

    def __init__(self, A: int, T: int, D: int, S: int, N: int, size_threshold: int) -> None:
        """Initialize the MotionDetection class with specified hyperparameters.

        Args:
            A (int): Frame hysteresis.
            T (int): Motion threshold.
            D (int): Distance threshold.
            S (int): Frame skip.
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

    def compute_motion(self) -> np.ndarray:
        """Compute the motion frame based on differences between consecutive frames.

        Returns:
            np.ndarray: The motion frame computed from the current frame buffer.
        """
        frame1, frame2, frame3 = self.frame_buffer
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)
        motion_frame = cv2.bitwise_and(diff1, diff2)
        _, thresh = cv2.threshold(motion_frame, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return thresh

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

    def reset_objects(self) -> None:
        """Reset the object tracking.

        This method clears the current list of tracked objects, effectively reinitializing
        the object tracking state. Used for starting a new tracking session or
        when the scene significantly changes.
        """
        self.tracked_objects = []


class TrackedObject:
    """Represents a single tracked object in the motion detection system.

    This class keeps track of an object's state using a Kalman filter, and stores its
    history of positions and the number of frames it has been inactive.
    """

    def __init__(self, filter: KalmanFilter, inactive_frames: int) -> None:
        """Initialize a TrackedObject with a Kalman filter and an inactive frame counter.

        Args:
            filter (KalmanFilter): The Kalman filter associated with the tracked object.
            inactive_frames (int): The number of frames for which the object has been inactive.
        """
        self.filter = filter
        self.inactive_frames = inactive_frames
        # Initialize an empty list to store past positions
        self.previous_positions = []


class QtDemo(QWidget):
    """A PyQt5 widget for demonstrating video processing and object tracking.

    This class provides a user interface for navigating through video frames,
    controlling video playback, and visualizing the results of motion detection
    and object tracking
    """

    def __init__(self, frames: list[np.ndarray]) -> None:
        """Initialize the QtDemo class for displaying video frames and tracking objects.

        Args:
            frames (list[np.ndarray]): A list of video frames to be processed and displayed.
        """
        super().__init__()
        self.frames = frames
        self.current_frame = 0

        # Initialize the motion detector
        self.motion_detector = MotionDetection(A=5, T=10, D=50, S=2, N=13, size_threshold=10)
        self.motion_detector.initialize(self.frames[:3])

        # UI Components
        self.button_next = QPushButton("Next Frame")
        self.button_back = QPushButton("Back 60 Frames")
        self.button_forward = QPushButton("Forward 60 Frames")
        self.img_label = QLabel(alignment=Qt.AlignCenter)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.setup_ui()

        # Connect functions
        self.button_next.clicked.connect(self.on_next_frame)
        self.button_back.clicked.connect(self.on_back_frames)
        self.button_forward.clicked.connect(self.on_forward_frames)
        self.frame_slider.sliderMoved.connect(self.on_slider_move)

    def setup_ui(self) -> None:
        """Configure the UI elements of the application."""
        h, w, c = self.frames[0].shape
        img = self.convert_frame_to_qimage(self.frames[0], w, h, c)
        self.img_label.setPixmap(QPixmap.fromImage(img))

        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.frames) - 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.img_label)
        layout.addWidget(self.button_next)
        layout.addWidget(self.button_back)
        layout.addWidget(self.button_forward)
        layout.addWidget(self.frame_slider)

        self.button_play = QPushButton("Play Video")
        self.button_stop = QPushButton("Stop Video")
        self.button_stop.setEnabled(False)

        layout.addWidget(self.button_play)
        layout.addWidget(self.button_stop)

        # Connect new buttons to their functions
        self.button_play.clicked.connect(self.play_video)
        self.button_stop.clicked.connect(self.stop_video)

        # Video playback control attributes
        self.playback_active = False

    def play_video(self) -> None:
        """Start the video playback."""
        self.playback_active = True
        self.button_play.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.video_thread = Thread(target=self.update_video_frame, daemon=True)
        self.video_thread.start()

    def stop_video(self) -> None:
        """Stop the video playback."""
        self.playback_active = False
        self.button_play.setEnabled(True)
        self.button_stop.setEnabled(False)

    def update_video_frame(self) -> None:
        """Continuously update the video frame during playback.

        This method is intended to be run in a separate thread for video playback.
        It increments the current frame and updates the frame display.
        """
        while self.playback_active and self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.update_frame()
            time.sleep(1 / 30.0)  # Assuming 30 FPS

    def convert_frame_to_qimage(self, frame: np.ndarray, w: int, h: int, c: int) -> QImage:
        """Convert a video frame to a QImage object.

        Args:
            frame (np.ndarray): The video frame to be converted.
            w (int): The width of the frame.
            h (int): The height of the frame.
            c (int): The number of channels in the frame.

        Returns:
            QImage: The QImage object representing the video frame.
        """
        if c == 1:
            return QImage(frame, w, h, QImage.Format_Grayscale8)
        return QImage(frame, w, h, QImage.Format_RGB888)

    def draw_tracked_objects(self, frame: np.ndarray, w: int, h: int, c: int) -> QImage:
        """Draw tracked objects and their trails on the frame.

        Args:
            frame (np.ndarray): The current video frame.
            w (int): The width of the frame.
            h (int): The height of the frame.
            c (int): The number of channels in the frame.

        Returns:
            QImage: The QImage object with drawn tracked objects.
        """
        for obj in self.motion_detector.tracked_objects:
            # Draw the current position
            x, y = int(obj.filter.x[0, 0]), int(obj.filter.x[1, 0])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw the trail
            for pos in obj.previous_positions:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 2, (0, 255, 0), -1)

        return self.convert_frame_to_qimage(frame, w, h, c)

    @Slot()
    def on_next_frame(self) -> None:
        """Handle the event for the next frame button."""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.update_frame()

    @Slot()
    def on_back_frames(self) -> None:
        """Handle the event for the back 60 frames button."""
        previous_frame = self.current_frame
        self.current_frame = max(0, self.current_frame - 60)
        skips = previous_frame - self.current_frame
        self.update_frame_with_skips(skips)

    @Slot()
    def on_forward_frames(self) -> None:
        """Handle the event for the forward 60 frames button."""
        previous_frame = self.current_frame
        self.current_frame = min(len(self.frames) - 1, self.current_frame + 60)
        skips = self.current_frame - previous_frame
        self.update_frame_with_skips(skips)

    @Slot()
    def on_slider_move(self, pos: int) -> None:
        """Handle the slider move event.

        Args:
            pos (int): The new position of the slider.
        """
        previous_frame = self.current_frame
        self.current_frame = pos
        skips = abs(self.current_frame - previous_frame)
        self.update_frame_with_skips(skips)

    def update_frame(self) -> None:
        """Update the current frame and redraw tracked objects."""
        frame = self.frames[self.current_frame].copy()
        h, w, c = frame.shape
        self.motion_detector.update(frame)

        img = self.draw_tracked_objects(frame, w, h, c)
        self.img_label.setPixmap(QPixmap.fromImage(img))
        self.frame_slider.setValue(self.current_frame)
        # Reinitialize the motion detector if the user revisits a frame
        if self.current_frame < len(self.motion_detector.frame_buffer):
            self.motion_detector.initialize(self.frames[: self.current_frame + 1])

    def update_frame_with_skips(self, skips: int) -> None:
        """Update the frame considering the number of frame skips.

        Args:
            skips (int): The number of frames skipped.
        """
        frame = self.frames[self.current_frame].copy()
        h, w, c = frame.shape

        # Large skip, process the new frame independently
        if skips > 3:
            self.motion_detector.reset_objects()
            self.motion_detector.update(frame)
        else:
            # Smaller skips, update as usual
            self.motion_detector.update(frame)

        img = self.draw_tracked_objects(frame, w, h, c)
        self.img_label.setPixmap(QPixmap.fromImage(img))
        self.frame_slider.setValue(self.current_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar="PATH_TO_VIDEO", type=str)
    parser.add_argument("--num_frames", metavar="n", type=int, default=-1)
    parser.add_argument("--grey", metavar="True/False", type=str, default=False)
    args = parser.parse_args()

    num_frames = args.num_frames

    if num_frames > 0:
        frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey)
    else:
        frames = vread(args.video_path, as_grey=args.grey)

    app = QApplication([])

    widget = QtDemo(frames)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
