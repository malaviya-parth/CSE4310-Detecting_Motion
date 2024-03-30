"""Initialization file for the detecting_motion package."""

__all__ = [
    "ArgparseLogger",
    "KalmanFilter",
    "MotionDetection",
    "TrackedObject",
    "setup_custom_logger",
]

from detecting_motion.tools.custom_log import ArgparseLogger, setup_custom_logger
from detecting_motion.tools.motion_util import KalmanFilter, MotionDetection, TrackedObject
