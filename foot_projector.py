import os
import numpy as np
from typing import Optional, List
import utils

PIXEL_TO_METER = 0.000264583

class FootProjector:
    def __init__(self, output_dir: str):
        self.__output_dir = output_dir
        self.__lead_file = os.path.join(output_dir, 'lead_smoothed.json')
        self.__follow_file = os.path.join(output_dir, 'follow_smoothed.json')
        self.__camera_tracking_file = os.path.join(output_dir, 'camera_tracking.json')
        self.__initial_camera_pose_file = os.path.join(output_dir, 'initial_camera_pose.json')
        self.__initial_camera_pose = utils.load_json(self.__initial_camera_pose_file)
        self.__initial_camera_position = np.array(self.__initial_camera_pose['position'])
        self.__image_size = (1920, 1080)  # Assuming HD resolution, adjust if needed
        
        # Extract camera data
        camera_tracking = utils.load_json_integer_keys(self.__camera_tracking_file)
        # Store quaternions
        self.__camera_quats = [np.array(frame['rotation']) for frame in camera_tracking.values()]
        self.__focal_lengths = [frame['focal_length'] for frame in camera_tracking.values()]

    @staticmethod
    def quat_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [x,y,z,w] to 3x3 rotation matrix"""
        x, y, z, w = q
        
        x2, y2, z2 = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, yz, xz = x*y, y*z, x*z

        return np.array([
            [1 - 2*(y2 + z2), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(x2 + z2), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(x2 + y2)]
        ])

    def forward(self, frame: int) -> np.ndarray:
        """Get forward vector for given frame"""
        rotation_matrix = self.quat_to_matrix(self.__camera_quats[frame])
        return np.dot(rotation_matrix, np.array([0, 0, 1]))

    def project_point(self, img_point: np.ndarray, frame: int) -> np.ndarray:
        """Project 2D image point to 3D space"""
        # Rescale point to meters
        rescaled_pt = np.array([
            (img_point[0] - self.__image_size[0] / 2) * PIXEL_TO_METER,
            -(img_point[1] - self.__image_size[1] / 2) * PIXEL_TO_METER,  # flip Y
            0
        ])
        
        return self.adjust_points([rescaled_pt], frame)[0]

    def adjust_points(self, keypoints: List[np.ndarray], frame: int) -> List[np.ndarray]:
        """Adjust points based on camera position and rotation"""
        rotation_matrix = self.quat_to_matrix(self.__camera_quats[frame])
        focal_length = self.__focal_lengths[frame]

        # Translate to camera center
        adjusted = [pt + self.__initial_camera_position for pt in keypoints]
        
        # Rotate around camera center
        adjusted = [np.dot(rotation_matrix, (pt - self.__initial_camera_position)) + self.__initial_camera_position for pt in adjusted]
        
        # Translate to focal length
        forward = self.forward(frame)
        adjusted = [pt + forward * focal_length for pt in adjusted]
        
        return adjusted

    @staticmethod
    def ray_plane_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
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

    def img_pt_ray_floor_intersection(self, img_pt: np.ndarray, frame: int) -> Optional[np.ndarray]:
        """Project image point to floor"""
        projected_point = self.project_point(img_pt, frame)
        ray_direction = projected_point - self.__initial_camera_position
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        return self.ray_plane_intersection(self.__initial_camera_position, ray_direction)

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
                        floor_pos = self.img_pt_ray_floor_intersection(np.array(ankle_pos), int(frame))
                        if floor_pos is not None:
                            frame_ankles[ankle_name] = floor_pos.tolist()

            # Process follow ankles
            if frame in follow_poses and follow_poses[frame]['id'] != -1:
                for ankle_name, ankle_idx in [('follow_left', 15), ('follow_right', 16)]:
                    ankle_pos = follow_poses[frame]['keypoints'][ankle_idx][:2]
                    if ankle_pos[0] != 0 or ankle_pos[1] != 0:
                        floor_pos = self.img_pt_ray_floor_intersection(np.array(ankle_pos), int(frame))
                        if floor_pos is not None:
                            frame_ankles[ankle_name] = floor_pos.tolist()

            all_ankle_positions_per_frame[frame] = frame_ankles

        # Save results
        utils.save_json(all_ankle_positions_per_frame, os.path.join(self.__output_dir, 'all_floor_ankles.json'))
