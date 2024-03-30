import numpy as np

from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square


class MotionDetector:

    def __init__(self, A, T, D, S, N, size_threshold):
        self.A = A  # Frame hysteresis for determining active or inactive objects.
        self.T = T  # The motion threshold for filtering out noise.
        self.D = D  # A distance threshold to determine if an object candidate belongs to an object currently being tracked.
        self.S = S  # The number of frames to skip between detections. The tracker will still work well even if it is not updated every frame.
        self.N = N  # The maximum number of objects to track.
        self.size_threshold = size_threshold  # Adding the size threshold
        self.frames = []
        self.tracked_objects = []
        self.active_objects = []

    def update(self, new_frame):
        print("Updating the motion detector...")
        print(f"Updating motion detector for frame {len(self.frames)}")
        print(f"Frames: {len(self.frames)}")
        if len(self.frames) < 3:  # Collecting the first 3 frames
            self.frames.append(new_frame)
            return

        # Process the frame
        print("Processing the frame...")
        frame_diff1 = np.abs(self.frames[-1] - new_frame)
        frame_diff2 = np.abs(self.frames[-2] - self.frames[-1])
        motion_frame = np.minimum(frame_diff1, frame_diff2)
        motion_frame[motion_frame < self.T] = 0
        print(f"Frame: {motion_frame}, Detected motions: {len(self.active_objects)}")
        for obj in self.active_objects:
            print(f"Object at {obj['centroid']} with active count {obj['active_count']}")

        print("Checking the shape of the motion frame...")
        if motion_frame.ndim == 3 and motion_frame.shape[2] == 3:
            # Convert RGB to grayscale
            print("Converting RGB to grayscale...")
            motion_frame = rgb2gray(motion_frame)
        else:
            print("No need to convert RGB to grayscale...")

        # Dilation
        dilated_frame = dilation(motion_frame, square(9))

        # Blob detection
        print("Blob detection...")
        labeled_frame = label(dilated_frame)
        for region in regionprops(labeled_frame):
            # Size filtering
            if region.area >= self.size_threshold:
                # Object candidate found
                print(f"Object candidate found: {region}")
                self.active_objects.append(
                    {"centroid": region.centroid, "bbox": region.bbox, "active_count": 1}
                )

        print(f"Detected active objects: {len(self.active_objects)}")  # Debug print

        # Update tracked objects
        self.match_and_track()
        print(f"Currently tracking {len(self.tracked_objects)} objects")
        for tracked in self.tracked_objects:
            print(
                f"Object at {tracked['kalman_filter'].x} with velocity {tracked['kalman_filter'].x[1]}, {tracked['kalman_filter'].x[3]}"
            )

        # Update frames
        print("Updating frames...")
        if len(self.frames) > self.S:
            print("Popping the first frame...")
            self.frames.pop(0)
        self.frames.append(new_frame)

        print("End of Motion Detector Update...")

    def match_and_track(self):
        # # This function will be used to match object candidates with existing tracked objects
        # # and update their states or create new objects.

        # # Check if there are any tracked objects
        # if not self.tracked_objects:
        #     print("No tracked objects to match with. Adding new objects if any.")
        #     # If there are no tracked objects, consider adding new objects
        #     for candidate in self.active_objects:
        #         # Check for adding new tracked object
        #         if candidate["active_count"] >= self.A:
        #             new_kalman_filter = KalmanFilter(
        #                 dt=1, u_x=0, u_y=0, std_acc=1, x_std_meas=1, y_std_meas=1
        #             )
        #             new_kalman_filter.x[:2] = np.array(candidate["centroid"]).reshape(2, 1)
        #             self.tracked_objects.append(
        #                 {
        #                     "kalman_filter": new_kalman_filter,
        #                     "active_count": 0,
        #                     "positions": [candidate["centroid"]],
        #                 }
        #             )

        # # Example implementation:
        # # Use centroid distance to match objects and then update or create Kalman filters
        # # Predict new positions for each tracked object
        # for obj in self.tracked_objects:
        #     obj["kalman_filter"].predict()

        # # For each active object candidate, check if it matches a tracked object
        # for candidate in self.active_objects:
        #     candidate_position = np.array(candidate["centroid"]).reshape(2, 1)
        #     print(f"Candidate Position: {candidate_position}")
        #     distances = [
        #         np.linalg.norm(obj["kalman_filter"].x[:2] - candidate_position)
        #         for obj in self.tracked_objects
        #     ]

        #     # Check if candidate is close enough to a predicted position
        #     print(f"Distances: {distances}")
        #     if distances:
        #         min_distance = min(distances)
        #         if min_distance < self.D:
        #             index = distances.index(min_distance)
        #             # Update the Kalman filter with the new measurement
        #             self.tracked_objects[index]["kalman_filter"].update(candidate_position)
        #             # Reset the active count
        #             self.tracked_objects[index]["active_count"] = 0
        #         else:
        #             # If no tracked object is close enough, consider adding the candidate as a new object
        #             candidate["active_count"] += 1
        #             # If active for A frames, add as a new tracked object
        #             # if candidate["active_count"] >= self.A:
        #             #     new_kalman_filter = KalmanFilter(
        #             #         dt=1, u_x=0, u_y=0, std_acc=1, x_std_meas=1, y_std_meas=1
        #             #     )
        #             #     # Initialize the filter state with the current position and zero velocity
        #             #     new_kalman_filter.x[:2] = candidate_position
        #             #     self.tracked_objects.append(
        #             #         {
        #             #             "kalman_filter": new_kalman_filter,
        #             #             "active_count": 0,
        #             #             "positions": [candidate["centroid"]],  # Store the initial position
        #             #         }
        #             #     )
        #             # Check for adding new tracked object
        #             if candidate["active_count"] >= self.A:
        #                 new_kalman_filter = KalmanFilter(
        #                     dt=1, u_x=0, u_y=0, std_acc=1, x_std_meas=1, y_std_meas=1
        #                 )
        #                 new_kalman_filter.x[:2] = np.array(candidate["centroid"]).reshape(2, 1)
        #                 self.tracked_objects.append(
        #                     {
        #                         "kalman_filter": new_kalman_filter,
        #                         "active_count": 0,
        #                         "positions": [candidate["centroid"]],
        #                     }
        #                 )

        # print(f"Motion - Number of tracked objects: {len(self.tracked_objects)}")  # Debug print

        # # Remove inactive objects and increment active counts
        # self.tracked_objects = [obj for obj in self.tracked_objects if obj["active_count"] < self.A]
        # for obj in self.tracked_objects:
        #     obj["active_count"] += 1

        # # If the number of objects exceeds N, keep only the N most recently updated
        # self.tracked_objects = sorted(
        #     self.tracked_objects, key=lambda x: x["active_count"], reverse=True
        # )[: self.N]
        # Predict new positions for each tracked object
        for obj in self.tracked_objects:
            obj["kalman_filter"].predict()

        # Iterate over a copy of the list since we might modify the original list
        for candidate in self.active_objects[:]:
            candidate_position = np.array(candidate["centroid"]).reshape(2, 1)
            distances = [
                np.linalg.norm(obj["kalman_filter"].x[:2] - candidate_position)
                for obj in self.tracked_objects
            ]

            # If there are distances to compare and if the closest one is within the threshold
            if distances and min(distances) < self.D:
                index = distances.index(min(distances))
                self.tracked_objects[index]["kalman_filter"].update(candidate_position)
                self.tracked_objects[index]["active_count"] = 0
            else:
                # Increase the active count or add as a new object
                candidate["active_count"] += 1
                if candidate["active_count"] >= self.A:
                    # Initialize a new Kalman filter for the candidate
                    new_kalman_filter = KalmanFilter(
                        dt=(1 / 30), u_x=0, u_y=0, std_acc=0.5, x_std_meas=2, y_std_meas=2
                    )
                    new_kalman_filter.x[:2] = candidate_position
                    new_kalman_filter.x[2:] = np.array([[0], [0]])  # Initialize velocity to 0

                    # Add the new object to the tracked objects
                    self.tracked_objects.append(
                        {
                            "kalman_filter": new_kalman_filter,
                            "active_count": 0,
                            "positions": [candidate["centroid"]],
                        }
                    )
                    self.active_objects.remove(
                        candidate
                    )  # Remove the candidate from active_objects


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        # Define the state transition matrix
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        # Define the control-input matrix
        self.B = np.array([[dt**2 / 2, 0], [dt, 0], [0, dt**2 / 2], [0, dt]])

        # Define the control-input (acceleration)
        self.u = np.array([u_x, u_y]).reshape((2, 1))

        # Define the measurement matrix
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # Initial covariance matrix
        self.P = np.eye(self.F.shape[1])

        self.G = np.array([[0.5 * dt**2], [dt], [0.5 * dt**2], [dt]])

        # Process noise matrix
        self.Q = self.G.dot(self.G.T) * std_acc**2

        # Measurement noise matrix
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])

        # The initial state (location and velocity)
        self.x = np.zeros((self.F.shape[1], 1))

        # The list of the object's previous positions
        self.positions = []

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        # Update the state with the measurement
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[1])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        # Store the position
        self.positions.append(self.x)
