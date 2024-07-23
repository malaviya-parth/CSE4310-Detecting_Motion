"""Module for motion detection and object tracking GUI functionality."""

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


class VideoPlayerGUI(QWidget):
    """A PyQt5 interface for video frame display and object tracking.

    This class provides controls for video playback, frame navigation, 
    and visualization of motion detection and object tracking results.
    """

    def __init__(self, video_frames: list[np.ndarray]) -> None:
        """Initialize the VideoPlayerGUI with the given video frames.

        Args:
            video_frames (list[np.ndarray]): A list of video frames to be processed and displayed.
        """
        super().__init__()
        self.video_frames = video_frames
        self.current_index = 0
        self.is_playing = False

        # Initialize the motion detector
        self.detector = MotionDetection(A=5, T=10, D=50, S=2, N=13, size_threshold=10)
        self.detector.initialize(self.video_frames[:3])

        # Setup the user interface
        self.setup_ui()

        # Apply custom stylesheet
        self.set_stylesheet()

    def setup_ui(self) -> None:
        """Setup the user interface components."""
        # Initialize control buttons
        self.btn_next = QPushButton("Next Frame")
        self.btn_back = QPushButton("Back 60 Frames")
        self.btn_forward = QPushButton("Forward 60 Frames")
        self.btn_play = QPushButton("Play Video")
        self.btn_stop = QPushButton("Stop Video")
        self.btn_stop.setEnabled(False)

        # Initialize the image display label
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        h, w, c = self.video_frames[0].shape
        img = self.convert_frame_to_qimage(self.video_frames[0], w, h, c)
        self.image_label.setPixmap(QPixmap.fromImage(img))

        # Initialize the slider for frame navigation
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.video_frames) - 1)

        # QTimer for video playback
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_frame)

        # Setup layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.frame_slider)

        # Setup control buttons layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_next)
        btn_layout.addWidget(self.btn_back)
        btn_layout.addWidget(self.btn_forward)
        layout.addLayout(btn_layout)

        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_stop)

        # Set the layout
        self.setLayout(layout)

        # Connect button signals to their respective slots
        self.connect_signals()

    def connect_signals(self) -> None:
        """Connect button signals to their respective slots."""
        self.btn_next.clicked.connect(self.on_next_frame)
        self.btn_back.clicked.connect(self.on_back_frames)
        self.btn_forward.clicked.connect(self.on_forward_frames)
        self.frame_slider.sliderMoved.connect(self.on_slider_move)
        self.btn_play.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_video)

    def set_stylesheet(self) -> None:
        """Apply custom stylesheet to the GUI components."""
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
    def convert_frame_to_qimage(frame: np.ndarray, width: int, height: int, channels: int) -> QImage:
        """Convert a video frame to a QImage object.

        Args:
            frame (np.ndarray): The video frame to be converted.
            width (int): The width of the frame.
            height (int): The height of the frame.
            channels (int): The number of channels in the frame.

        Returns:
            QImage: The QImage object representing the video frame.
        """
        if channels == 1:
            return QImage(frame, width, height, QImage.Format_Grayscale8)
        return QImage(frame, width, height, QImage.Format_RGB888)

    def render_tracked_objects(self, frame: np.ndarray, width: int, height: int, channels: int) -> QImage:
        """Draw tracked objects and their trails on the frame.

        Args:
            frame (np.ndarray): The current video frame.
            width (int): The width of the frame.
            height (int): The height of the frame.
            channels (int): The number of channels in the frame.

        Returns:
            QImage: The QImage object with drawn tracked objects.
        """
        for obj in self.detector.tracked_objects:
            # Draw the current position
            x, y = int(obj.filter.x[0, 0]), int(obj.filter.x[1, 0])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Draw the trail
            for pos in obj.previous_positions:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 2, (0, 255, 0), -1)

        return self.convert_frame_to_qimage(frame, width, height, channels)

    def update_frame(self) -> None:
        """Update the current frame and redraw tracked objects."""
        if self.current_index < len(self.video_frames) - 1:
            self.current_index += 1
            frame = self.video_frames[self.current_index].copy()
            h, w, c = frame.shape
            self.detector.update(frame)

            img = self.render_tracked_objects(frame, w, h, c)
            self.image_label.setPixmap(QPixmap.fromImage(img))
            self.frame_slider.setValue(self.current_index)
        else:
            # Stop the timer if it's the last frame
            self.playback_timer.stop()

        # Reinitialize the motion detector if the user revisits a frame
        if self.current_index < len(self.detector.frame_buffer):
            self.detector.initialize(self.video_frames[: self.current_index + 1])

    def update_frame_with_skips(self, skips: int) -> None:
        """Update the frame considering the number of frame skips.

        Args:
            skips (int): The number of frames skipped.
        """
        frame = self.video_frames[self.current_index].copy()
        h, w, c = frame.shape

        # Large skip, process the new frame independently
        if skips > 3:
            self.detector.update_with_skips(frame, skips)
        else:
            # Smaller skips, update as usual
            self.detector.update(frame)

        img = self.render_tracked_objects(frame, w, h, c)
        self.image_label.setPixmap(QPixmap.fromImage(img))
        self.frame_slider.setValue(self.current_index)

    def update_video_frame(self) -> None:
        """Continuously update the video frame during playback.

        It increments the current frame and updates the frame display.
        """
        while self.is_playing and self.current_index < len(self.video_frames) - 1:
            self.current_index += 1
            self.update_frame()
            time.sleep(1 / 30.0)  # Assuming 30 FPS

    @Slot()
    def on_next_frame(self) -> None:
        """Handle the event for the next frame button."""
        if self.current_index < len(self.video_frames) - 1:
            self.current_index += 1
            self.update_frame()

    @Slot()
    def on_back_frames(self) -> None:
        """Handle the event for the back 60 frames button."""
        previous_index = self.current_index
        self.current_index = max(0, self.current_index - 60)
        skips = previous_index - self.current_index
        self.update_frame_with_skips(skips)

    @Slot()
    def on_forward_frames(self) -> None:
        """Handle the event for the forward 60 frames button."""
        previous_index = self.current_index
        self.current_index = min(len(self.video_frames) - 1, self.current_index + 60)
        skips = self.current_index - previous_index
        self.update_frame_with_skips(skips)

    @Slot()
    def on_slider_move(self, pos: int) -> None:
        """Handle the slider move event.

        Args:
            pos (int): The new position of the slider.
        """
        previous_index = self.current_index
        self.current_index = pos
        skips = abs(self.current_index - previous_index)
        self.update_frame_with_skips(skips)

    @Slot()
    def start_video(self) -> None:
        """Start the video playback."""
        self.is_playing = True
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.playback_timer.start(33)  # Approx. 30 FPS

    @Slot()
    def stop_video(self) -> None:
        """Stop the video playback."""
        self.is_playing = False
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.playback_timer.stop()


def read_video_frames(file_path: str, frame_count: int = -1, grayscale: bool = False) -> list[np.ndarray]:
    """Load frames from a video file.

    Args:
        file_path (str): Path to the video file.
        frame_count (int): Number of frames to load. (Default: -1)
        grayscale (bool): Flag to load frames as grayscale. (Default: False)

    Returns:
        list[np.ndarray]: List of video frames.
    """
    try:
        skvideo.setFFmpegPath("./detecting_motion/ffmpeg_essentials_build/bin")
        from skvideo.io import vread  # noqa: PLC0415

        if frame_count > 0:
            frames = vread(file_path, num_frames=frame_count, as_grey=grayscale)
        else:
            frames = vread(file_path, as_grey=grayscale)
    except AssertionError:
        try:
            if frame_count > 0:
                frames = vread(file_path, num_frames=frame_count, as_grey=grayscale)
            else:
                frames = vread(file_path, as_grey=grayscale)
        except AssertionError:
            logging.exception("FFmpeg path invalid or not found. Check the path and try again.")
            sys.exit(1)

    return frames


def parse_arguments(logger: Logger) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = ArgparseLogger(
        logger,
        description="GUI for video frame loading and object tracking.",
    )
    parser.add_argument(
        "file_path", metavar="VIDEO_FILE_PATH", type=str, help="Path to the video file."
    )
    parser.add_argument(
        "-n",
        "--frame_count",
        metavar="frame_count",
        type=int,
        default=-1,
        help="Number of frames to load. (Default: -1)",
    )
    parser.add_argument(
        "-g",
        "--grayscale",
        metavar="True/False",
        type=bool,
        default=False,
        help="Flag to load frames as grayscale. (Default: False)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run the GUI."""
    logger = setup_logger()

    args = parse_arguments(logger)

    frame_count = args.frame_count

    video_frames = read_video_frames(args.file_path, frame_count=frame_count, grayscale=args.grayscale)

    app = QApplication([])

    widget = VideoPlayerGUI(video_frames)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
