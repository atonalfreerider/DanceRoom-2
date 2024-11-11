import os
import argparse
from pathlib import Path
import copy
import numpy as np

import utils

def find_lowest_y(pose_frame):
    """Find the lowest y value in the pose"""
    return min(joint['y'] for joint in pose_frame)

def find_floor_foot(pose_frame):
    """Find which foot (L or R) is closest to the ground and return its info"""
    l_foot_y = pose_frame[10]['y']  # L_Foot
    r_foot_y = pose_frame[11]['y']  # R_Foot
    
    if l_foot_y <= r_foot_y:
        return 'left', pose_frame[7]  # L_Ankle
    else:
        return 'right', pose_frame[8]  # R_Ankle

def translate_pose(pose_frame, translation):
    """Translate all joints in a pose by a given vector"""
    translated_frame = copy.deepcopy(pose_frame)
    for joint in translated_frame:
        joint['x'] += translation[0]
        joint['y'] += translation[1]
        joint['z'] += translation[2]
    return translated_frame

def ground_pose(pose_frame):
    """Translate pose so lowest joint is at y=0"""
    # TODO this prohibits any pose from jumping through the air. In the future, actual jump detection will be required
    lowest_y = find_lowest_y(pose_frame)
    return translate_pose(pose_frame, [0, -lowest_y, 0])

def get_pose_center(pose_frame):
    """Get the average position of all joints as the pose center"""
    x_sum = sum(joint['x'] for joint in pose_frame)
    z_sum = sum(joint['z'] for joint in pose_frame)
    n_joints = len(pose_frame)
    return x_sum/n_joints, z_sum/n_joints

def detect_discontinuity(prev_frame, curr_frame, threshold=0.25):
    """Detect if there's a large position change between frames"""
    if prev_frame is None or curr_frame is None:
        return False
        
    prev_center = get_pose_center(prev_frame)
    curr_center = get_pose_center(curr_frame)
    
    distance = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                      (prev_center[1] - curr_center[1])**2)
    
    return distance > threshold

def align_pose_to_floor_xz(pose_frame, floor_ankle_pos, floor_ankle):
    """Align pose so floor ankle matches floor position in x/z plane only"""
    translation = [
        floor_ankle_pos[0] - floor_ankle['x'],
        0,  # No y translation
        floor_ankle_pos[2] - floor_ankle['z']
    ]
    return translate_pose(pose_frame, translation)

def realign_at_discontinuity(frame, ground_ankles, frame_idx, role):
    """Realign pose using floor ankle data at the current frame"""
    floor_side, floor_ankle = find_floor_foot(frame)
    floor_key = f'{role}_{floor_side}'
    
    if str(frame_idx) not in ground_ankles or floor_key not in ground_ankles[str(frame_idx)]:
        return frame
        
    floor_ankle_pos = ground_ankles[str(frame_idx)][floor_key]
    return align_pose_to_floor_xz(frame, floor_ankle_pos, floor_ankle)

def process_frames_with_discontinuity_correction(frames, ground_ankles, role):
    """Process all frames with initial alignment and discontinuity correction"""
    if not frames:
        return []
    
    # First ground all poses
    processed_frames = [ground_pose(frame) for frame in frames]
    
    # Initial alignment using frame 0
    frame_0_floor_pos = ground_ankles.get('0', {})
    floor_side, floor_ankle = find_floor_foot(processed_frames[0])
    floor_key = f'{role}_{floor_side}'
    
    if floor_key in frame_0_floor_pos:
        floor_ankle_pos = frame_0_floor_pos[floor_key]
        initial_translation = [
            floor_ankle_pos[0] - floor_ankle['x'],
            0,
            floor_ankle_pos[2] - floor_ankle['z']
        ]
        # Apply initial translation to all frames
        processed_frames = [translate_pose(frame, initial_translation) for frame in processed_frames]
    
    # Process remaining frames with discontinuity detection
    for i in range(1, len(processed_frames)):
        if detect_discontinuity(processed_frames[i-1], processed_frames[i]):
            print(f"Detected discontinuity for {role} at frame {i}")
            
            # Calculate correction translation using floor ankle data
            floor_side, floor_ankle = find_floor_foot(processed_frames[i])
            floor_key = f'{role}_{floor_side}'
            
            if str(i) in ground_ankles and floor_key in ground_ankles[str(i)]:
                floor_ankle_pos = ground_ankles[str(i)][floor_key]
                correction_translation = [
                    floor_ankle_pos[0] - floor_ankle['x'],
                    0,
                    floor_ankle_pos[2] - floor_ankle['z']
                ]
                
                # Apply correction translation to all frames from i onwards
                for j in range(i, len(processed_frames)):
                    processed_frames[j] = translate_pose(processed_frames[j], correction_translation)
    
    return processed_frames

def get_height(pose_frame):
    """Calculate height from legs and spine up to neck"""
    # Left leg
    l_thigh = np.array([pose_frame[4]['x'] - pose_frame[1]['x'],
                       pose_frame[4]['y'] - pose_frame[1]['y'],
                       pose_frame[4]['z'] - pose_frame[1]['z']])
    l_calf = np.array([pose_frame[7]['x'] - pose_frame[4]['x'],
                      pose_frame[7]['y'] - pose_frame[4]['y'],
                      pose_frame[7]['z'] - pose_frame[4]['z']])
    
    # Right leg
    r_thigh = np.array([pose_frame[5]['x'] - pose_frame[2]['x'],
                       pose_frame[5]['y'] - pose_frame[2]['y'],
                       pose_frame[5]['z'] - pose_frame[2]['z']])
    r_calf = np.array([pose_frame[8]['x'] - pose_frame[5]['x'],
                      pose_frame[8]['y'] - pose_frame[5]['y'],
                      pose_frame[8]['z'] - pose_frame[5]['z']])
    
    # Spine segments
    spine1 = np.array([pose_frame[3]['x'] - pose_frame[0]['x'],
                      pose_frame[3]['y'] - pose_frame[0]['y'],
                      pose_frame[3]['z'] - pose_frame[0]['z']])
    spine2 = np.array([pose_frame[6]['x'] - pose_frame[3]['x'],
                      pose_frame[6]['y'] - pose_frame[3]['y'],
                      pose_frame[6]['z'] - pose_frame[3]['z']])
    spine3 = np.array([pose_frame[9]['x'] - pose_frame[6]['x'],
                      pose_frame[9]['y'] - pose_frame[6]['y'],
                      pose_frame[9]['z'] - pose_frame[6]['z']])
    neck = np.array([pose_frame[12]['x'] - pose_frame[9]['x'],
                    pose_frame[12]['y'] - pose_frame[9]['y'],
                    pose_frame[12]['z'] - pose_frame[9]['z']])
    
    # Calculate lengths
    l_leg_length = np.linalg.norm(l_thigh) + np.linalg.norm(l_calf)
    r_leg_length = np.linalg.norm(r_thigh) + np.linalg.norm(r_calf)
    spine_length = (np.linalg.norm(spine1) + np.linalg.norm(spine2) + 
                   np.linalg.norm(spine3) + np.linalg.norm(neck))
    
    # Use average of left and right leg
    return (l_leg_length + r_leg_length) / 2 + spine_length

def get_animation_center(frames):
    """Get the center point of the entire animation"""
    all_x = []
    all_z = []
    for frame in frames:
        for joint in frame:
            all_x.append(joint['x'])
            all_z.append(joint['z'])
    
    return np.mean(all_x), 0, np.mean(all_z)

def scale_animation(frames, scale_factor):
    """Scale entire animation uniformly"""
    if not frames:
        return frames
    
    scaled_frames = []
    for frame in frames:
        scaled_frame = copy.deepcopy(frame)
        for joint in scaled_frame:
            # Scale all coordinates
            joint['x'] *= scale_factor
            joint['y'] *= scale_factor
            joint['z'] *= scale_factor
        
        scaled_frames.append(scaled_frame)
    
    return scaled_frames

def scale_poses_to_match_ratio(lead_frames, follow_frames, target_ratio):
    """Scale follow animation to match target height ratio with lead"""
    if not lead_frames or not follow_frames:
        return follow_frames
    
    # Calculate initial heights from frame 0
    lead_height = get_height(lead_frames[0])
    follow_height = get_height(follow_frames[0])
    
    # Calculate current ratio and needed scale factor
    current_ratio = follow_height / lead_height
    scale_factor = target_ratio / current_ratio
    
    print(f"Current height ratio: {current_ratio:.3f}")
    print(f"Applying scale factor: {scale_factor:.3f} to match target ratio: {target_ratio:.3f}")
    
    # Scale entire follow animation
    scaled_follow_frames = scale_animation(follow_frames, scale_factor)
    
    return scaled_follow_frames

def main(output_dir:str, lead_follow_height_ratio:float):
    ground_ankle_path = os.path.join(output_dir, 'all_floor_ankles.json')
    ground_ankles = utils.load_json(ground_ankle_path)

    lead_keypoints_path = os.path.join(output_dir, 'lead_smoothed_keypoints_3d.json')
    lead_keypoints = utils.load_json(lead_keypoints_path)

    follow_keypoints_path = os.path.join(output_dir, 'follow_smoothed_keypoints_3d.json')
    follow_keypoints = utils.load_json(follow_keypoints_path)

    # First scale the follow poses to match height ratio
    follow_keypoints = scale_poses_to_match_ratio(lead_keypoints, follow_keypoints, lead_follow_height_ratio)

    # Then process both dancers with discontinuity correction
    aligned_lead_keypoints = process_frames_with_discontinuity_correction(
        lead_keypoints, ground_ankles, 'lead')
    aligned_follow_keypoints = process_frames_with_discontinuity_correction(
        follow_keypoints, ground_ankles, 'follow')

    # Save aligned poses
    lead_output_path = os.path.join(output_dir, 'lead_aligned_keypoints_3d.json')
    follow_output_path = os.path.join(output_dir, 'follow_aligned_keypoints_3d.json')
    utils.save_json(aligned_lead_keypoints, lead_output_path)
    utils.save_json(aligned_follow_keypoints, follow_output_path)

    dir_path = Path(output_dir)
    dir_name = dir_path.stem

    rhythm_path = os.path.join(output_dir, f'{dir_name}_zouk-time-analysis.json')
    rhythm = utils.load_json(rhythm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align 3D poses, and compute rhythm physics, patterns.")
    parser.add_argument("--output_dir", help="Path to the output directory")
    parser.add_argument("--lead_follow_height_ratio", type=float, default=0.875,
                      help="Target ratio of follow height to lead height")
    args = parser.parse_args()

    main(args.output_dir, args.lead_follow_height_ratio)
