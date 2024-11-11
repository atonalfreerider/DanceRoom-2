import cv2
import numpy as np
from collections import OrderedDict

import utils


class PoseDataUtils:
    @staticmethod
    def create_empty_pose():
        return {
            "id": -1,
            "bbox": [0, 0, 0, 0],
            "confidence": 0,
            "keypoints": [[0, 0, 0] for _ in range(17)]
        }

    @staticmethod
    def validate_pose(pose):
        required_keys = ["id", "bbox", "confidence", "keypoints"]
        for key in required_keys:
            if key not in pose:
                raise ValueError(f"Missing required key: {key}")
        
        if len(pose["bbox"]) != 4:
            raise ValueError("bbox must have exactly 4 elements")
        
        if len(pose["keypoints"]) != 17:
            raise ValueError("keypoints must have exactly 17 elements")
        
        for keypoint in pose["keypoints"]:
            if len(keypoint) != 3:
                raise ValueError("Each keypoint must have exactly 3 elements")

    @staticmethod
    def format_poses(poses_dict, num_frames):
        formatted_poses = OrderedDict()
        
        for frame in range(num_frames):
            if frame in poses_dict and poses_dict[frame]:
                formatted_poses[frame] = poses_dict[frame]
                PoseDataUtils.validate_pose(formatted_poses[frame])
            else:
                formatted_poses[frame] = PoseDataUtils.create_empty_pose()
        
        return formatted_poses

    @staticmethod
    def save_poses(poses_dict, num_frames, output_file):
        if not poses_dict:
            print(f"Warning: Poses dictionary is empty. Not saving to {output_file}")
            return

        formatted_poses = PoseDataUtils.format_poses(poses_dict, num_frames)
        
        # Check if the formatted poses are not just empty frames
        if all(len(poses) == 1 and poses[0]['id'] == -1 for poses in formatted_poses.values()):
            print(f"Warning: All poses are empty. Not saving to {output_file}")
            return

        utils.save_json(formatted_poses, output_file)

    @staticmethod
    def draw_pose(image, keypoints, track_id, pose_type):
        color = (125, 125, 125)
        if pose_type == 'lead':
            color = (0, 0, 255)
        elif pose_type == 'follow':
            color = (255, 192, 203)
        elif pose_type == 'hovered':
            color = (255, 255, 0)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        overlay = np.zeros_like(image, dtype=np.uint8)

        def is_valid_point(point):
            return point[0] != 0 or point[1] != 0

        def get_point_and_conf(kp):
            if len(kp) == 4 and (kp[0] != 0 or kp[1] != 0):
                return (int(kp[0]), int(kp[1])), kp[3]  # x, y, z, conf
            elif len(kp) == 3 and (kp[0] != 0 or kp[1] != 0):
                return (int(kp[0]), int(kp[1])), kp[2]  # x, y, conf
            return None, 0.0

        line_thickness = 1 if pose_type == 'unknown' else 3

        for connection in connections:
            if len(keypoints) > max(connection):
                start_point, start_conf = get_point_and_conf(keypoints[connection[0]])
                end_point, end_conf = get_point_and_conf(keypoints[connection[1]])

                if start_point is not None and end_point is not None:
                    avg_conf = (start_conf + end_conf) / 2
                    color_with_alpha = tuple(int(c * avg_conf) for c in color)
                    cv2.line(overlay, start_point, end_point, color_with_alpha, line_thickness)

        for point in keypoints:
            pt, conf = get_point_and_conf(point)
            if pt is not None:
                color_with_alpha = tuple(int(c * conf) for c in color)
                cv2.circle(overlay, pt, 3, color_with_alpha, -1)

        if pose_type == 'lead' or pose_type == 'follow':
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

        cv2.add(image, overlay, image)

        text = str(track_id)
        if pose_type == 'lead':
            text = 'LEAD ' + text
        elif pose_type == 'follow':
            text = 'FOLLOW ' + text

        cv2.putText(image, text, (int(keypoints[0][0]), int(keypoints[0][1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)