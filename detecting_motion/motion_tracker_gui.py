"""This module provides the main functionality for motion detection and object tracking."""

import argparse
import logging
import sys
import time

from logging import Logger

import cv2
import numpy as np
import skvideo

from PySide2.QtCore import Qt, QTimer, Slot
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from detecting_motion import ArgparseLogger, MotionDetection, setup_custom_logger


np.float = np.float64
np.int = np.int_


class GUI(QWidget):
    """A PyQt5 widget for demonstrating video processing and object tracking.

    This class provides a user interface for navigating through video frames,
    controlling video playback, and visualizing the results of motion detection
    and object tracking
    """

    def __init__(self, frames: list[np.ndarray]) -> None:
        """Initialize the GUI class for displaying video frames and tracking objects.

        Args:
            frames (list[np.ndarray]): A list of video frames to be processed and displayed.
        """
        super().__init__()
        self.frames = frames
        self.current_frame = 0

        # Video playback control attribute
        self.playback_active = False

        # Initialize the motion detector
        self.motion_detector = MotionDetection(A=5, T=10, D=50, S=2, N=13, size_threshold=10)
        self.motion_detector.initialize(self.frames[:3])

        # Initialize UI components
        self.init_ui()

        # Set up the stylesheet
        self.apply_stylesheet()

    def init_ui(self) -> None:
        """Initialize the user interface"""
        # Initialize Buttons
        self.button_next = QPushButton("Next Frame")
        self.button_back = QPushButton("Back 60 Frames")
        self.button_forward = QPushButton("Forward 60 Frames")
        self.button_play = QPushButton("Play Video")
        self.button_stop = QPushButton("Stop Video")
        self.button_stop.setEnabled(False)

        # Initialize the image label
        self.img_label = QLabel(alignment=Qt.AlignCenter)
        h, w, c = self.frames[0].shape
        img = self.frame_to_qimage(self.frames[0], w, h, c)
        self.img_label.setPixmap(QPixmap.fromImage(img))

        # Initialize the slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.frames) - 1)

        # QTimer for video playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Set up the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.img_label)
        layout.addWidget(self.frame_slider)

        # Set up the Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_next)
        button_layout.addWidget(self.button_back)
        button_layout.addWidget(self.button_forward)
        layout.addLayout(button_layout)

        layout.addWidget(self.button_play)
        layout.addWidget(self.button_stop)

        # Set layout
        self.setLayout(layout)

        # Connect functions
        self.connect_signals()

    def connect_signals(self) -> None:
        """Connect signals to the corresponding slots."""
        self.button_next.clicked.connect(self.on_next_frame)
        self.button_back.clicked.connect(self.on_back_frames)
        self.button_forward.clicked.connect(self.on_forward_frames)
        self.frame_slider.sliderMoved.connect(self.on_slider_move)
        self.button_play.clicked.connect(self.play_video)
        self.button_stop.clicked.connect(self.stop_video)

    def apply_stylesheet(self) -> None:
        """Apply the stylesheet to the GUI components."""
        self.setStyleSheet(
            """
            QWidget {
                background-color: #EEEEEE;
                font-family: Arial;
            }
            QPushButton {
                background-color: #31363F;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #76ABAE;
            }
            QLabel {
                color: #222831;
            }
            QSlider {
                background: #222831;
            }
        """
        )

    @staticmethod
    def frame_to_qimage(frame: np.ndarray, w: int, h: int, c: int) -> QImage:
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

        return self.frame_to_qimage(frame, w, h, c)

    def update_frame(self) -> None:
        """Update the current frame and redraw tracked objects."""
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            frame = self.frames[self.current_frame].copy()
            h, w, c = frame.shape
            self.motion_detector.update(frame)

            img = self.draw_tracked_objects(frame, w, h, c)
            self.img_label.setPixmap(QPixmap.fromImage(img))
            self.frame_slider.setValue(self.current_frame)
        else:
            # Stop the timer if it's the last frame
            self.timer.stop()

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
            self.motion_detector.update_with_skips(frame, skips)
        else:
            # Smaller skips, update as usual
            self.motion_detector.update(frame)

        img = self.draw_tracked_objects(frame, w, h, c)
        self.img_label.setPixmap(QPixmap.fromImage(img))
        self.frame_slider.setValue(self.current_frame)

    def update_video_frame(self) -> None:
        """Continuously update the video frame during playback.

        It increments the current frame and updates the frame display.
        """
        while self.playback_active and self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.update_frame()
            time.sleep(1 / 30.0)  # Assuming 30 FPS

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

    @Slot()
    def play_video(self) -> None:
        """Start the video playback."""
        self.playback_active = True
        self.button_play.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.timer.start(33)  # Start the timer with interval in ms (33ms for ~30FPS)

    @Slot()
    def stop_video(self) -> None:
        """Stop the video playback."""
        self.playback_active = False
        self.button_play.setEnabled(True)
        self.button_stop.setEnabled(False)
        self.timer.stop()  # Stop the timer


def load_frames(video_path: str, num_frames: int = -1, grey: bool = False) -> list[np.ndarray]:
    """Load the video frames from a video file.

    Args:
        video_path (str): The path to the video file.
        num_frames (int): The number of frames to load. (Default: -1)
        grey (bool): Flag to indicate if frames should be loaded as grayscale. (Default: False)

    Returns:
        list[np.ndarray]: A list of video frames loaded from the video file.
    """
    try:
        # Set the FFmpeg path for skvideo
        skvideo.setFFmpegPath("./detecting_motion/ffmpeg_essentials_build/bin")
        from skvideo.io import vread  # noqa: PLC0415

        if num_frames > 0:
            frames = vread(video_path, num_frames=num_frames, as_grey=grey)
        else:
            frames = vread(video_path, as_grey=grey)
    except AssertionError:
        # If the FFmpeg path does not get set, try loading the video with user's FFmpeg
        try:
            if num_frames > 0:
                frames = vread(video_path, num_frames=num_frames, as_grey=grey)
            else:
                frames = vread(video_path, as_grey=grey)
        except AssertionError:
            logging.exception("FFmpeg path invalid or not found. Check the path and try again.")
            sys.exit(1)

    return frames


def _parse_args(custom_logger: Logger) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = ArgparseLogger(
        custom_logger,
        description="GUI for loading and displaying video frames with object tracking.",
    )
    parser.add_argument(
        "video_path", metavar="PATH_TO_VIDEO", type=str, help="Path to the video file."
    )
    parser.add_argument(
        "-n",
        "--num_frames",
        metavar="num_frames",
        type=int,
        default=-1,
        help="Number of frames to load. (Default: -1)",
    )
    parser.add_argument(
        "-g",
        "--grey",
        metavar="True/False",
        type=bool,
        default=False,
        help="Flag to indicate if frames should be loaded as grayscale. (Default: False)",
    )
    return parser.parse_args()


def _main() -> None:
    """Main function for running the GUI."""
    custom_logger = setup_custom_logger()

    args = _parse_args(custom_logger)

    num_frames = args.num_frames

    frames = load_frames(args.video_path, num_frames=num_frames, grey=args.grey)

    app = QApplication([])

    widget = GUI(frames)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    _main()
