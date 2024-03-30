import argparse
import random
import sys

import cv2
import numpy as np
import skvideo

from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square

from detecting_motion import KalmanFilter, MotionDetector


# Set the path to the FFmpeg binary, which is required by skvideo
#   -- Set a try and except later so that it can be used on different systems if installed
skvideo.setFFmpegPath(
    "C:\\Users\\Administrator\\Desktop\\School\\3_Detecting_Motion\\ffmpeg_essentials_build\\bin"
)

from skvideo.io import vread


np.float = np.float64
np.int = np.int_


class KalmanFilter:
    def __init__(self):
        # Initializing the Kalman Filter parameters
        # These parameters can be adjusted based on the specific tracking scenario
        self.dt = 1  # Time step
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # State transition matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
        self.Q = np.eye(4) * 0.1  # Process noise covariance
        self.R = np.eye(2) * 20  # Measurement noise covariance
        self.P = np.eye(4) * 500  # Estimate error covariance
        self.x = np.zeros((4, 1))  # Initial state (x, y, vx, vy)

    def predict(self):
        # Prediction step of the Kalman Filter
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        # Update step of the Kalman Filter
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x += np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(
            np.dot(K, self.R), K.T
        )


class MotionDetection:

    def __init__(self, A, T, D, S, N, size_threshold):
        # Hyperparameters
        self.frame_hysteresis = A
        self.motion_threshold = T
        self.distance_threshold = D
        self.frame_skip = S
        self.max_objects = N
        self.size_threshold = size_threshold

        # State variables
        self.frame_buffer = []
        self.tracked_objects = []

    def initialize(self, frames):
        # Initialize the detector with the first three frames
        if len(frames) < 3:
            raise ValueError("At least 3 frames are needed for initialization")
        for frame in frames[:3]:
            self.update(frame, initialize=True)

    def update(self, frame, initialize=False):
        # Update the frame buffer
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

    def compute_motion(self):
        # Compute the motion frame based on differences between consecutive frames
        frame1, frame2, frame3 = self.frame_buffer
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)
        motion_frame = cv2.bitwise_and(diff1, diff2)
        _, thresh = cv2.threshold(motion_frame, self.motion_threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def detect_objects(self, motion_frame):
        # Ensure motion_frame is 2D
        if len(motion_frame.shape) == 3:
            motion_frame = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)

        # Detect objects in the motion frame
        dilated_frame = dilation(motion_frame, square(3))
        labeled_frame = label(dilated_frame)
        objects = regionprops(labeled_frame)
        return [obj for obj in objects if obj.area >= self.size_threshold]

    def update_tracked_objects(self, detected_objects):
        # Update the list of tracked objects
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
            # Optionally, limit the size of the trail history
            obj.previous_positions = obj.previous_positions[
                -10:
            ]  # Keep last 10 positions, for example

        # Remove inactive objects
        self.tracked_objects = [
            obj for obj in self.tracked_objects if obj.inactive_frames <= self.frame_hysteresis
        ]
        for obj in self.tracked_objects:
            obj.inactive_frames += 1
            obj.filter.predict()


class TrackedObject:
    def __init__(self, filter, inactive_frames):
        self.filter = filter
        self.inactive_frames = inactive_frames
        self.previous_positions = []  # Initialize an empty list to store past positions


class QtDemo(QWidget):  # Updated QtDemo class with motion detection and tracking
    def __init__(self, frames) -> None:
        super().__init__()
        self.frames = frames
        self.current_frame = 0

        # Initialize the motion detector
        self.motion_detector = MotionDetection(A=3, T=15, D=20, S=1, N=10, size_threshold=100)
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

    def setup_ui(self):
        # Configure the UI elements
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

    def convert_frame_to_qimage(self, frame, w, h, c):
        # Convert a frame to QImage
        if c == 1:
            return QImage(frame, w, h, QImage.Format_Grayscale8)
        else:
            return QImage(frame, w, h, QImage.Format_RGB888)

    def draw_tracked_objects(self, frame, w, h, c):
        # Draw tracked objects and their trails on the frame
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
        # Handle the next frame button
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.update_frame()

    @Slot()
    def on_back_frames(self) -> None:
        # Handle the back 60 frames button
        self.current_frame = max(0, self.current_frame - 60)
        self.update_frame()

    @Slot()
    def on_forward_frames(self) -> None:
        # Handle the forward 60 frames button
        self.current_frame = min(len(self.frames) - 1, self.current_frame + 60)
        self.update_frame()

    @Slot()
    def on_slider_move(self, pos) -> None:
        # Handle the slider move
        self.current_frame = pos
        self.update_frame()

    def update_frame(self):
        # Update the current frame and redraw tracked objects
        frame = self.frames[self.current_frame].copy()
        h, w, c = frame.shape
        self.motion_detector.update(frame)

        img = self.draw_tracked_objects(frame, w, h, c)
        self.img_label.setPixmap(QPixmap.fromImage(img))
        self.frame_slider.setValue(self.current_frame)
        # Reinitialize the motion detector if the user revisits a frame
        if self.current_frame < len(self.motion_detector.frame_buffer):
            self.motion_detector.initialize(self.frames[: self.current_frame + 1])


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
