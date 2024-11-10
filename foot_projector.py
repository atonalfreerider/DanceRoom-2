import os
import numpy as np
from typing import Optional, List
from scipy.spatial.transform import Rotation
import utils

class FootProjector:
    def __init__(self, output_dir: str, frame_height: int, frame_width: int):
        self.__output_dir = output_dir
        self.__lead_file = os.path.join(output_dir, 'lead_smoothed.json')
        self.__follow_file = os.path.join(output_dir, 'follow_smoothed.json')
        self.__camera_tracking_file = os.path.join(output_dir, 'camera_tracking.json')
        self.__initial_camera_pose = utils.load_json(os.path.join(output_dir, 'initial_camera_pose.json'))
        self.__initial_camera_position = np.array(self.__initial_camera_pose['position'])
        self.__frame_height = frame_height
        self.__frame_width = frame_width
        
        # Extract camera data
        camera_tracking = utils.load_json_integer_keys(self.__camera_tracking_file)
        self.__camera_quats = [np.array(frame['rotation']) for frame in camera_tracking.values()]
        self.__focal_lengths = [frame['focal_length'] for frame in camera_tracking.values()]

    def __project_point_to_floor(self, pixel_x: int, pixel_y: int, frame: int) -> Optional[np.ndarray]:
        """Project a 2D image point to the floor plane using the same logic as virtual_room"""
        # Get camera parameters
        fx = fy = min(self.__frame_height, self.__frame_width)
        cx, cy = self.__frame_width / 2, self.__frame_height / 2
        focal_length = self.__focal_lengths[frame]
        rotation = Rotation.from_quat(self.__camera_quats[frame])

        # Convert to normalized device coordinates (same as virtual_room)
        x_ndc = -(pixel_x - cx) / fx  # Flip X to match world space
        y_ndc = (cy - pixel_y) / fy   # Flip Y to match world space

        # Create ray in camera space
        ray_dir = np.array([
            x_ndc / focal_length,
            y_ndc / focal_length,
            1.0
        ])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Transform ray to world space
        world_ray = rotation.apply(ray_dir)

        # Calculate intersection with floor (y = 0)
        if abs(world_ray[1]) > 1e-6:  # Check if ray is not parallel to floor
            t = -self.__initial_camera_position[1] / world_ray[1]
            if t > 0:  # Check if intersection is in front of camera
                intersection = self.__initial_camera_position + t * world_ray
                return intersection

        return None

    def project_feet_to_ground(self):
        # Load data
        lead_poses = utils.load_json_integer_keys(self.__lead_file)
        follow_poses = utils.load_json_integer_keys(self.__follow_file)

        all_ankle_positions_per_frame = {}

        # Process each frame
        for frame in lead_poses.keys():
            frame_ankles = {}
            
            # Process lead ankles (indices 15 and 16 are left and right ankles)
            if lead_poses[frame]['id'] != -1:
                for ankle_name, ankle_idx in [('lead_left', 15), ('lead_right', 16)]:
                    ankle_pos = lead_poses[frame]['keypoints'][ankle_idx][:2]  # Get x,y coordinates
                    if ankle_pos[0] != 0 or ankle_pos[1] != 0:  # Check if valid keypoint
                        floor_pos = self.__project_point_to_floor(
                            int(ankle_pos[0]), 
                            int(ankle_pos[1]), 
                            int(frame)
                        )
                        if floor_pos is not None:
                            frame_ankles[ankle_name] = floor_pos.tolist()

            # Process follow ankles
            if frame in follow_poses and follow_poses[frame]['id'] != -1:
                for ankle_name, ankle_idx in [('follow_left', 15), ('follow_right', 16)]:
                    ankle_pos = follow_poses[frame]['keypoints'][ankle_idx][:2]
                    if ankle_pos[0] != 0 or ankle_pos[1] != 0:
                        floor_pos = self.__project_point_to_floor(
                            int(ankle_pos[0]), 
                            int(ankle_pos[1]), 
                            int(frame)
                        )
                        if floor_pos is not None:
                            frame_ankles[ankle_name] = floor_pos.tolist()

            all_ankle_positions_per_frame[frame] = frame_ankles

        # Save results
        utils.save_json(all_ankle_positions_per_frame, os.path.join(self.__output_dir, 'all_floor_ankles.json'))
