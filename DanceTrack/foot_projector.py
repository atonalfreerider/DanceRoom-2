import os
import numpy as np
from typing import Optional, Dict, List, Tuple
import utils
from scipy.spatial.transform import Rotation
from sklearn.linear_model import RANSACRegressor


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

    @staticmethod
    def __find_streaks(poses: Dict, frame_indices: List[int]) -> List[Tuple[str, List[int]]]:
        """Find consecutive sequences where one foot is lower than the other"""
        streaks = []
        current_foot = None
        current_streak = []
        
        for frame in frame_indices:
            if poses[frame]['id'] == -1:
                if current_streak:
                    streaks.append((current_foot, current_streak))
                    current_streak = []
                continue
                
            left_y = poses[frame]['keypoints'][15][1]  # y-coordinate
            right_y = poses[frame]['keypoints'][16][1]
            
            if left_y <= 0 and right_y <= 0:
                if current_streak:
                    streaks.append((current_foot, current_streak))
                    current_streak = []
                continue
                
            foot = 'left' if left_y > right_y and left_y > 0 else 'right' if right_y > 0 else None
            
            if foot != current_foot and current_streak:
                streaks.append((current_foot, current_streak))
                current_streak = []
            
            if foot is not None:
                current_foot = foot
                current_streak.append(frame)
                
        if current_streak:
            streaks.append((current_foot, current_streak))
            
        return streaks

    @staticmethod
    def __clean_floor_points(points: List[np.ndarray]) -> Optional[np.ndarray]:
        """Apply RANSAC and average to a sequence of floor points"""
        if len(points) < 3:  # Need at least 3 points for meaningful RANSAC
            return np.mean(points, axis=0) if points else None
            
        points = np.array(points)
        X = points[:, [0, 2]]  # Use X and Z coordinates for RANSAC
        y = points[:, 1]  # Y coordinates
        
        ransac = RANSACRegressor(random_state=42, min_samples=3)
        try:
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            cleaned_points = points[inlier_mask]
            return np.mean(cleaned_points, axis=0) if len(cleaned_points) > 0 else None
        except:
            return np.mean(points, axis=0) if len(points) > 0 else None

    def project_feet_to_ground(self):
        # Load data
        lead_poses = utils.load_json_integer_keys(self.__lead_file)
        follow_poses = utils.load_json_integer_keys(self.__follow_file)
        
        # Get frame indices
        frame_indices = sorted(lead_poses.keys())
        
        # Find streaks for both dancers
        lead_streaks = self.__find_streaks(lead_poses, frame_indices)
        follow_streaks = self.__find_streaks(follow_poses, frame_indices)
        
        # Process streaks into floor positions
        all_ankle_positions_per_frame = {}
        
        # Process each streak for lead and follow
        for is_lead, streaks in [(True, lead_streaks), (False, follow_streaks)]:
            dancer_prefix = 'lead' if is_lead else 'follow'
            poses = lead_poses if is_lead else follow_poses
            
            for foot, streak_frames in streaks:
                floor_points = []
                
                # Collect floor points for the streak
                for frame in streak_frames:
                    focal_length = self.__focal_lengths[frame]
                    rotation = self.__camera_quats[frame]
                    bbox = poses[frame]['bbox']
                    y2 = bbox[3]
                    
                    if y2 > self.__frame_height * .95:
                        continue
                        
                    keypoint_idx = 15 if foot == 'left' else 16
                    ankle_pos = poses[frame]['keypoints'][keypoint_idx][:2]
                    
                    if ankle_pos[1] > 0:
                        floor_pos = self.__project_point_to_floor(
                            [ankle_pos[0], y2], rotation, focal_length)
                        if floor_pos is not None:
                            floor_points.append(floor_pos)
                
                # Clean and average the floor points
                if floor_points:
                    cleaned_pos = self.__clean_floor_points(floor_points)
                    if cleaned_pos is not None:
                        # Apply the cleaned position to all frames in the streak
                        for frame in streak_frames:
                            if frame not in all_ankle_positions_per_frame:
                                all_ankle_positions_per_frame[frame] = {}
                            key = f'{dancer_prefix}_{foot}'
                            all_ankle_positions_per_frame[frame][key] = cleaned_pos

        # Save results
        utils.save_numpy_json(all_ankle_positions_per_frame, 
                            os.path.join(self.__output_dir, 'all_floor_ankles.json'))
