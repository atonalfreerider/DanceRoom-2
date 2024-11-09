import os
from pathlib import Path
from pose_data_utils import PoseDataUtils
import utils

class TemporalSmoothing:
    def __init__(self, output_dir:str):
        self.output_dir = Path(output_dir)
        self.lead_file = os.path.join(output_dir, 'lead-normalized.json')
        self.follow_file = os.path.join(output_dir, 'follow-normalized.json')
        self.smoothed_lead_file = os.path.join(output_dir, 'lead_smoothed.json')
        self.smoothed_follow_file = os.path.join(output_dir, 'follow_smoothed.json')

    @staticmethod
    def is_valid_keypoint(keypoint):
        return keypoint[0] != 0 or keypoint[1] != 0

    @staticmethod
    def is_empty_pose(pose):
        return all(kp == [0, 0, 0] for kp in pose['keypoints'])

    def interpolate_missing_poses(self, role_data):
        interpolated_data = {}
        all_frames = sorted(role_data.keys())

        for i, frame in enumerate(all_frames):
            if frame in role_data and not self.is_empty_pose(role_data[frame]):
                interpolated_data[frame] = role_data[frame]
            else:
                # Find the nearest valid poses before and after the current frame
                prev_valid_frame = next((f for f in reversed(all_frames[:i]) if f in role_data and not self.is_empty_pose(role_data[f])), None)
                next_valid_frame = next((f for f in all_frames[i+1:] if f in role_data and not self.is_empty_pose(role_data[f])), None)

                if prev_valid_frame is not None and next_valid_frame is not None:
                    prev_pose = role_data[prev_valid_frame]
                    next_pose = role_data[next_valid_frame]
                    
                    interpolated_pose = prev_pose.copy()
                    interpolated_keypoints = []

                    for j in range(17):  # 17 keypoints in COCO format
                        prev_kp = prev_pose['keypoints'][j]
                        next_kp = next_pose['keypoints'][j]

                        if self.is_valid_keypoint(prev_kp) and self.is_valid_keypoint(next_kp):
                            t = (frame - prev_valid_frame) / (next_valid_frame - prev_valid_frame)
                            interpolated_x = prev_kp[0] + t * (next_kp[0] - prev_kp[0])
                            interpolated_y = prev_kp[1] + t * (next_kp[1] - prev_kp[1])
                            interpolated_conf = prev_kp[2] + t * (next_kp[2] - prev_kp[2])
                            interpolated_keypoints.append([interpolated_x, interpolated_y, interpolated_conf])
                        else:
                            interpolated_keypoints.append([0, 0, 0])

                    interpolated_pose['keypoints'] = interpolated_keypoints
                    interpolated_data[frame] = interpolated_pose

        return interpolated_data

    def run(self):
        lead_data = utils.load_json_integer_keys(self.lead_file)
        follow_data = utils.load_json_integer_keys(self.follow_file)

        interpolated_lead = self.interpolate_missing_poses(lead_data)
        interpolated_follow = self.interpolate_missing_poses(follow_data)

        PoseDataUtils.save_poses(interpolated_lead, max(lead_data.keys()) + 1, self.smoothed_lead_file)
        PoseDataUtils.save_poses(interpolated_follow, max(follow_data.keys()) + 1, self.smoothed_follow_file)

        print(f"Interpolated lead data saved to: {self.smoothed_lead_file}")
        print(f"Interpolated follow data saved to: {self.smoothed_follow_file}")
