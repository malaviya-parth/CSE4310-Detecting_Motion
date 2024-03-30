import argparse
import random
import sys

import numpy as np
import skvideo

from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from detecting_motion import KalmanFilter, MotionDetector


# Set the path to the FFmpeg binary, which is required by skvideo
#   -- Set a try and except later so that it can be used on different systems if installed
skvideo.setFFmpegPath(
    "C:\\Users\\Administrator\\Desktop\\School\\3_Detecting_Motion\\ffmpeg_essentials_build\\bin"
)

from skvideo.io import vread  # noqa: E402


np.float = np.float64
np.int = np.int_


class QtDemo(QWidget):  # noqa: D101
    def __init__(self, frames) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001, D107
        super().__init__()
        self.frames = frames
        self.current_frame = 0
        self.motion_detector = MotionDetector(A=4, T=20, D=20, S=1, N=10, size_threshold=400)

        self.button = QPushButton("Next Frame")

        # Configure image label
        self.img_label = QLabel(alignment=Qt.AlignCenter)  # type: ignore[call-overload]
        h, w, c = self.frames[0].shape
        if c == 1:
            img = QImage(self.frames[0], w, h, QImage.Format_Grayscale8)
        else:
            img = QImage(self.frames[0], w, h, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(img))

        # Configure slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0] - 1)

        self.layout = QVBoxLayout(self)  # type: ignore[method-assign]
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.frame_slider)

        # Add jump buttons
        self.button_back_60 = QPushButton("Back 60 Frames")
        self.button_forward_60 = QPushButton("Forward 60 Frames")
        self.layout.addWidget(self.button_back_60)
        self.layout.addWidget(self.button_forward_60)

        # Connect functions
        self.button.clicked.connect(self.on_click)
        self.frame_slider.sliderMoved.connect(self.on_move)
        self.button_back_60.clicked.connect(self.jump_back)
        self.button_forward_60.clicked.connect(self.jump_forward)

    @Slot()  # type: ignore[operator]
    def on_click(self) -> None:
        if self.current_frame == self.frames.shape[0] - 1:
            return

        # Update the motion detector for the next frame
        self.update_motion_detector()
        self.current_frame += 1
        self.frame_slider.setValue(self.current_frame)

    @Slot()  # type: ignore[operator]
    def on_move(self, pos) -> None:
        self.current_frame = pos
        self.update_motion_detector()
        self.frame_slider.setValue(self.current_frame)

    @Slot()
    def jump_back(self):
        self.current_frame = max(self.current_frame - 60, 0)
        self.update_motion_detector()
        self.frame_slider.setValue(self.current_frame)

    @Slot()
    def jump_forward(self):
        self.current_frame = min(self.current_frame + 60, self.frames.shape[0] - 1)
        self.update_motion_detector()
        self.frame_slider.setValue(self.current_frame)

    def update_motion_detector(self):
        # We're moving to a new frame, either by clicking the next frame button or moving the slider
        # Check if we've already seen this frame. If we have, we need to re-initialize our motion detector
        # up to this frame, otherwise, we can simply update it with the new frame
        if self.current_frame < len(self.motion_detector.frames):
            self.motion_detector = MotionDetector(A=4, T=20, D=20, S=1, N=10, size_threshold=400)
        # Initialize or update the motion detector with the current frame
        print("Checking the shape of the motion frame before passed to 'motion_detector.update'...")
        if (
            self.frames[self.current_frame].ndim == 3
            and self.frames[self.current_frame].shape[2] == 3
        ):
            print("RGB Image")
        else:
            print("Greyscale Image")

        self.motion_detector.update(self.frames[self.current_frame])

        # Convert the current frame to QImage
        qimg = self.convert_frame_to_qimage(self.frames[self.current_frame])
        # Draw the tracked objects on the QImage
        qimg_with_trails = self.draw_tracked_objects(qimg)
        # Update the label with the new QPixmap
        self.img_label.setPixmap(QPixmap.fromImage(qimg_with_trails))

    def convert_frame_to_qimage(self, frame):
        # This function converts a NumPy array frame to a QImage
        if len(frame.shape) == 3:
            h, w, c = frame.shape
            bytes_per_line = 3 * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = frame.shape
            qimg = QImage(frame.data, w, h, QImage.Format_Grayscale8)
        # qimg = qimg.copy()  # Make a copy to avoid modifying the original frame
        return qimg

    def draw_tracked_objects(self, qimg):
        # Ensure that we're working with a copy of the QImage to avoid modifying the original
        painter = QPainter()

        qimg_copy = qimg.copy()  # Make a copy to avoid modifying the original frame
        painter.begin(qimg_copy)

        # Choose the pen color and width
        pen = QPen(QColor(255, 0, 0))  # red color
        pen.setWidth(2)
        painter.setPen(pen)

        # Debugging: Print the number of tracked objects
        print(f"Draw - Number of tracked objects: {len(self.motion_detector.tracked_objects)}")
        for obj in self.motion_detector.tracked_objects:
            # Draw the trail of detections
            for pos in obj["kalman_filter"].positions:
                x, y = int(pos[0][0]), int(pos[2][0])
                painter.drawPoint(x, y)

            # Draw the current position with a larger dot or a different shape
            current_position = obj["kalman_filter"].x
            x, y = int(current_position[0][0]), int(current_position[2][0])
            painter.drawEllipse(x - 5, y - 5, 10, 10)  # draw a circle for the current position

        painter.end()
        # Debugging: Print the number of tracked objects
        print(
            f"Draw - Number of tracked objects at painter end: {len(self.motion_detector.tracked_objects)}"
        )
        return qimg_copy  # Return the modified QImage


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
