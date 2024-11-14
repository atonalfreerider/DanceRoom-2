import os
import numpy as np
from typing import Optional
import utils
from scipy.spatial.transform import Rotation


class FootProjector:
    def __init__(self, output_dir: str, frame_height: int, frame_width: int):
        self.__output_dir = output_dir
        self.__lead_file = os.path.join(output_dir, 'lead.json')
        self.__follow_file = os.path.join(output_dir, 'follow.json')
        self.__initial_camera_pose = utils.load_json(os.path.join(output_dir, 'initial_camera_pose.json'))
        self.__initial_camera_position = np.array(self.__initial_camera_pose['position'])
        self.__frame_height, self.__frame_width = frame_height, frame_width

        # Extract camera data
        camera_tracking = utils.load_json_integer_keys(os.path.join(output_dir, 'camera_tracking.json'))
        # Store quaternions
        self.__camera_quats = [np.array(frame['rotation']) for frame in camera_tracking.values()]
        self.__focal_lengths = [frame['focal_length'] for frame in camera_tracking.values()]

    def __project_point_to_floor(self, image_point, rotation_quat, focal_length):
        """Project image point to world coordinates and determine which plane it lies on"""
        fx = fy = min(self.__frame_height, self.__frame_width)
        cx, cy = self.__frame_width / 2, self.__frame_height / 2
        rotation = Rotation.from_quat(rotation_quat)

        # Convert to normalized device coordinates - flip X sign to change orientation
        x_ndc = -(image_point[0] - cx) / fx  # Negative sign to flip X orientation
        y_ndc = (cy - image_point[1]) / fy  # Keep Y flipped

        # Create ray in camera space (positive Z is forward)
        ray_dir = np.array([
            x_ndc / focal_length,
            y_ndc / focal_length,
            1.0  # Positive Z for forward
        ])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Transform ray to world space
        world_ray = rotation.apply(ray_dir)

        return self.__ray_floor_intersection(self.__initial_camera_position, world_ray)

    @staticmethod
    def __ray_floor_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculate intersection of ray with floor plane (Y=0)"""
        plane_normal = np.array([0, 1, 0])  # Y-up
        plane_d = 0  # Floor at Y=0

        denominator = np.dot(plane_normal, ray_direction)

        if abs(denominator) <= np.finfo(float).eps:
            return None

        t = (-plane_d - np.dot(plane_normal, ray_origin)) / denominator

        if t < 0:
            return None

        intersection = ray_origin + t * ray_direction
        return intersection

    def project_feet_to_ground(self):
        # Load data
        lead_poses = utils.load_json_integer_keys(self.__lead_file)
        follow_poses = utils.load_json_integer_keys(self.__follow_file)

        all_ankle_positions_per_frame = {}

        # Process each frame
        for frame in lead_poses.keys():
            frame_ankles = {}
            focal_length = self.__focal_lengths[frame]
            rotation = self.__camera_quats[frame]

            # Process lead ankles (indices 15 and 16 are left and right ankles)
            if lead_poses[frame]['id'] != -1:
                lead_bbox = lead_poses[frame]['bbox']
                lead_y2 = lead_bbox[3]
                if lead_y2 > self.__frame_height  *.95:
                    continue # bbox bottom is too close to bottom of frame -> skip

                lead_left = lead_poses[frame]['keypoints'][15][:2]
                lead_right = lead_poses[frame]['keypoints'][16][:2]

                if lead_left[1] > lead_right[1] and lead_left[1] > 0:
                    floor_pos = self.__project_point_to_floor([lead_left[0], lead_y2], rotation, focal_length)
                    frame_ankles['lead_left'] = floor_pos
                elif lead_right[1] > 0:
                    floor_pos = self.__project_point_to_floor([lead_right[0], lead_y2], rotation, focal_length)
                    frame_ankles['lead_right'] = floor_pos

            # Process follow ankles (indices 15 and 16 are left and right ankles)
            if follow_poses[frame]['id'] != -1:
                follow_bbox = follow_poses[frame]['bbox']
                follow_y2 = follow_bbox[3]
                if follow_y2 > self.__frame_height * .95:
                    continue  # bbox bottom is too close to bottom of frame -> skip

                follow_left = follow_poses[frame]['keypoints'][15][:2]
                follow_right = follow_poses[frame]['keypoints'][16][:2]

                if follow_left[1] > follow_right[1] and follow_left[1] > 0:
                    floor_pos = self.__project_point_to_floor([follow_left[0], follow_y2], rotation, focal_length)
                    frame_ankles['follow_left'] = floor_pos
                elif follow_right[1] > 0:
                    floor_pos = self.__project_point_to_floor([follow_right[0], follow_y2], rotation, focal_length)
                    frame_ankles['follow_right'] = floor_pos

            all_ankle_positions_per_frame[frame] = frame_ankles

        # Save results
        utils.save_numpy_json(all_ankle_positions_per_frame, os.path.join(self.__output_dir, 'all_floor_ankles.json'))
