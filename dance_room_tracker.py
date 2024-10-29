import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import utils
from cotracker_runner import CoTracker


class DanceRoomTracker:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.camera_poses_json_path = output_dir + '/camera_poses.json'
        self.camera_poses = self.load_camera_poses()
        self.cotracker = CoTracker()

    def load_camera_poses(self):
        """Load initial camera poses from JSON if it exists"""
        data = utils.load_json(self.camera_poses_json_path)
        if data:
            return {
                'position': np.array(data['position']),
                'rotation': np.array(data['rotation']),
                'focal_length': data['focal_length']
            }
        else:
            # Default camera pose
            return {
                'position': np.array([0.0, 1.2, 3.5]),
                'rotation': np.array([0.0, 0.0, 0.0, 1.0]),  # Forward-facing quaternion
                'focal_length': 0.35
            }

    def save_camera_poses(self):
        """Save camera poses to JSON"""
        data = {
            'position': self.camera_poses['position'].tolist(),
            'rotation': self.camera_poses['rotation'].tolist(),
            'focal_length': self.camera_poses['focal_length']
        }
        utils.save_json(data, self.camera_poses_json_path)

    def calibrate_camera(self):
        """Interactive camera calibration function"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        # Get video properties
        H, W = frame.shape[:2]

        # Camera movement parameters
        pos_delta = 0.1  # Translation step size
        rot_delta = 0.02  # Rotation step size
        focal_delta = 0.01  # Focal length step size

        def log_camera_state():
            """Helper function to log camera state"""
            pos = self.camera_poses['position']
            rot = self.camera_poses['rotation']
            print(f"\nCamera State:")
            print(f"Position (XYZ): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"Rotation (XYZW): [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f}]")
            print(f"Focal Length: {self.camera_poses['focal_length']:.3f}")

        calibration_done = False
        while not calibration_done:
            # Create copy of frame for drawing
            display_frame = frame.copy()

            # Calculate intrinsics based on frame size
            fx = fy = min(H, W)
            cx, cy = W/2, H/2
            intrinsics = np.array([fx, fy, cx, cy])

            # Draw origin overlay
            current_rotation = Rotation.from_quat(self.camera_poses['rotation'])
            display_frame = self.draw_origin_corner(
                display_frame,
                self.camera_poses['position'],
                current_rotation,
                self.camera_poses['focal_length'],
                intrinsics
            )

            # Show frame
            cv2.imshow('Camera Calibration', display_frame)
            key = cv2.waitKey(1) & 0xFF

            # Handle keyboard input
            update = True
            if key == ord('w'):  # Forward
                self.camera_poses['position'][2] -= pos_delta
            elif key == ord('s'):  # Backward
                self.camera_poses['position'][2] += pos_delta
            elif key == ord('a'):  # Left
                self.camera_poses['position'][0] -= pos_delta
            elif key == ord('d'):  # Right
                self.camera_poses['position'][0] += pos_delta
            elif key == ord('q'):  # Up
                self.camera_poses['position'][1] += pos_delta
            elif key == ord('e'):  # Down
                self.camera_poses['position'][1] -= pos_delta
            elif key == ord('z'):  # Increase focal length
                self.camera_poses['focal_length'] += focal_delta
            elif key == ord('x'):  # Decrease focal length
                self.camera_poses['focal_length'] = max(0.1, self.camera_poses['focal_length'] - focal_delta)
            elif key == 82:  # Up arrow (tilt up)
                rot = Rotation.from_euler('x', -rot_delta)
                self.camera_poses['rotation'] = (rot * Rotation.from_quat(self.camera_poses['rotation'])).as_quat()
            elif key == 84:  # Down arrow (tilt down)
                rot = Rotation.from_euler('x', rot_delta)
                self.camera_poses['rotation'] = (rot * Rotation.from_quat(self.camera_poses['rotation'])).as_quat()
            elif key == 81:  # Left arrow (pan left)
                rot = Rotation.from_euler('y', -rot_delta)
                self.camera_poses['rotation'] = (rot * Rotation.from_quat(self.camera_poses['rotation'])).as_quat()
            elif key == 83:  # Right arrow (pan right)
                rot = Rotation.from_euler('y', rot_delta)
                self.camera_poses['rotation'] = (rot * Rotation.from_quat(self.camera_poses['rotation'])).as_quat()
            elif key == ord('o'):  # Roll counter-clockwise
                rot = Rotation.from_euler('z', -rot_delta)
                self.camera_poses['rotation'] = (rot * Rotation.from_quat(self.camera_poses['rotation'])).as_quat()
            elif key == ord('p'):  # Roll clockwise
                rot = Rotation.from_euler('z', rot_delta)
                self.camera_poses['rotation'] = (rot * Rotation.from_quat(self.camera_poses['rotation'])).as_quat()
            elif key == 13:  # Enter key
                self.save_camera_poses()
                calibration_done = True
            elif key == 27:  # ESC key
                break
            else:
                update = False

            # Log camera state if there was an update
            if update:
                log_camera_state()

        cap.release()
        cv2.destroyAllWindows()

    def X(self):
        # TODO
        points = None
        pred_track, pred_vis = CoTracker.track(self.video_path, points)

    def render(self):
        """Render debug overlay video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

    def reverse_project_point(self, world_point, position, rotation, focal_length):
        target = (world_point - position) / np.linalg.norm(world_point - position)
        return self.get_image_plane_coordinates(target, rotation, focal_length)

    @staticmethod
    def get_image_plane_coordinates(ray_direction, rotation, focal_length):
        forward = rotation.apply([0, 0, 1])
        up = rotation.apply([0, 1, 0])
        right = rotation.apply([1, 0, 0])

        t = focal_length / np.dot(forward, ray_direction)
        intersection_point = t * ray_direction

        image_plane_coordinates = np.array([
            np.dot(intersection_point, right),
            np.dot(intersection_point, up)
        ])

        return image_plane_coordinates

    def draw_origin_corner(self, frame, camera_position, camera_rotation, focal_length, intrinsics):
        fx, fy, cx, cy = intrinsics

        # Define the origin point and axis endpoints (1 meter each)
        origin_points = np.array([
            [0, 0, 0],  # Origin
            [1, 0, 0],  # X-axis endpoint
            [0, 1, 0],  # Y-axis endpoint
            [0, 0, 1]   # Z-axis endpoint
        ])

        # Project points to image plane
        image_points = []
        for point in origin_points:
            img_coords = self.reverse_project_point(point, camera_position, camera_rotation, focal_length)
            pixel_x = int(img_coords[0] * fx + cx)
            pixel_y = int(img_coords[1] * fy + cy)
            image_points.append((pixel_x, pixel_y))

        # Draw the origin corner
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z
        for i, color in enumerate(colors):
            cv2.line(frame, image_points[0], image_points[i+1], color, 2)

        return frame

    @staticmethod
    def draw_pose(image, keypoints, color, is_lead_or_follow=False):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        def is_valid_point(point):
            return point[0] != 0 or point[1] != 0

        for connection in connections:
            start_point = keypoints[connection[0]][:2]
            end_point = keypoints[connection[1]][:2]
            if is_valid_point(start_point) and is_valid_point(end_point):
                cv2.line(image, tuple(map(int, start_point)), tuple(map(int, end_point)), color, 2)

        for point in keypoints:
            if is_valid_point(point[:2]):
                cv2.circle(image, tuple(map(int, point[:2])), 3, color, -1)

        if is_lead_or_follow:
            # Draw 'L' on left side
            left_shoulder = keypoints[5][:2]
            left_hip = keypoints[11][:2]
            if is_valid_point(left_shoulder) and is_valid_point(left_hip):
                mid_point = ((left_shoulder[0] + left_hip[0]) // 2, (left_shoulder[1] + left_hip[1]) // 2)
                cv2.putText(image, 'L', tuple(map(int, mid_point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw 'R' on right side
            right_shoulder = keypoints[6][:2]
            right_hip = keypoints[12][:2]
            if is_valid_point(right_shoulder) and is_valid_point(right_hip):
                mid_point = ((right_shoulder[0] + right_hip[0]) // 2, (right_shoulder[1] + right_hip[1]) // 2)
                cv2.putText(image, 'R', tuple(map(int, mid_point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

