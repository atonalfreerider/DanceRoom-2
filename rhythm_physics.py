import os
import argparse
from pathlib import Path
import copy
import numpy as np

import utils

def find_lowest_y(pose_frame):
    """Find the lowest y value in the pose"""
    return min(joint['y'] for joint in pose_frame)

def find_floor_foot(pose_frame, camera_position, force_ground=False):
    """Find which foot (L or R) is closest to the ground and closest to camera position, or None"""
    l_foot_y = pose_frame[10]['y']  # L_Foot
    r_foot_y = pose_frame[11]['y']  # R_Foot
    
    l_foot_dist = np.sqrt((pose_frame[10]['x'] - camera_position[0])**2 + 
                          (pose_frame[10]['z'] + camera_position[2])**2)  # Camera Z is opposite of GVHMR output
    r_foot_dist = np.sqrt((pose_frame[11]['x'] - camera_position[0])**2 + 
                          (pose_frame[11]['z'] + camera_position[2])**2)  # Camera Z is opposite of GVHMR output

    if l_foot_dist <= r_foot_dist:
        if l_foot_y < 0.15:
            return 'left', pose_frame[7]  # L_Ankle

    if r_foot_y < 0.15:
        return 'right', pose_frame[8]  # R_Ankle

    if force_ground:
        return None, None

    return 'left', pose_frame[7]  # L_Ankle

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

def process_frame_discontinuities(frames, ground_ankles, role, camera_position):
    """Process frames with initial alignment at discontinuities"""
    if not frames:
        return []
    
    # First ground all poses
    processed_frames = [ground_pose(frame) for frame in frames]
    
    # Initial alignment using frame 0
    frame_0_floor_pos = ground_ankles.get(0, {})
    floor_side, floor_ankle = find_floor_foot(processed_frames[0], camera_position)
    floor_key = f'{role}_{floor_side}'
    
    if floor_key not in frame_0_floor_pos:
        return processed_frames
        
    # Get frame 0 ground position
    frame_0_ground_pos = frame_0_floor_pos[floor_key]
    
    # Initial translation
    initial_translation = [
        frame_0_ground_pos[0] - floor_ankle['x'],
        0,
        -frame_0_ground_pos[2] - floor_ankle['z'] # ankle ground Z is opposite of GVHMR output
    ]
    processed_frames = [translate_pose(frame, initial_translation) for frame in processed_frames]
    
    # Find first discontinuity
    discontinuity_frame = None
    for i in range(1, len(processed_frames)):
        if detect_discontinuity(processed_frames[i-1], processed_frames[i]):
            print(f"Detected discontinuity for {role} at frame {i}")
            discontinuity_frame = i
            break
    
    if discontinuity_frame is not None:
        # Get floor ankle data at discontinuity
        floor_side, floor_ankle = find_floor_foot(processed_frames[discontinuity_frame], camera_position)
        floor_key = f'{role}_{floor_side}'
        
        if discontinuity_frame in ground_ankles and floor_key in ground_ankles[discontinuity_frame]:
            target_pos = ground_ankles[discontinuity_frame][floor_key]
            
            # Calculate translation needed at discontinuity point
            correction_translation = [
                target_pos[0] - floor_ankle['x'],
                0,
                -target_pos[2] - floor_ankle['z'] # ankle ground Z is opposite of GVHMR output
            ]
            
            # Apply translation to all frames from discontinuity onwards
            for j in range(discontinuity_frame, len(processed_frames)):
                processed_frames[j] = translate_pose(processed_frames[j], correction_translation)

    return processed_frames

def calculate_interpolation_factor(error_distance, max_error=0.25):
    """Calculate interpolation factor based on error distance"""
    return min(error_distance / max_error, 1.0)

def guide_poses_to_ground(frames, ground_ankles, role, camera_position):
    """Guide poses towards smoothed ground positions while maintaining continuity"""
    if not frames:
        return frames
    
    guided_frames = copy.deepcopy(frames)
    cumulative_translation = [0, 0, 0]  # Keep track of accumulated translation
    
    for i in range(len(guided_frames)):
        if i not in ground_ankles:
            continue

        # Apply cumulative translation to current frame
        guided_frames[i] = translate_pose(guided_frames[i], cumulative_translation)
            
        # Find grounded foot
        floor_side, floor_ankle = find_floor_foot(guided_frames[i], camera_position, force_ground=True)
        if floor_side is None:
            continue

        floor_key = f'{role}_{floor_side}'  # Use correct role
        
        if floor_key not in ground_ankles[i]:
            continue

        target_pos = ground_ankles[i][floor_key]
        
        # Calculate error and interpolation factor
        error_distance = np.sqrt(
            (floor_ankle['x'] - target_pos[0])**2 +
            (floor_ankle['z'] + target_pos[2])**2 # ankle ground Z is opposite of GVHMR output
        )
        
        interp_factor = calculate_interpolation_factor(error_distance)
        
        # Calculate frame translation
        frame_translation = [
            (target_pos[0] - floor_ankle['x']) * interp_factor,
            0,
            (-target_pos[2] - floor_ankle['z']) * interp_factor # ankle ground Z is opposite of GVHMR output
        ]
        
        # Update cumulative translation
        cumulative_translation = [
            cumulative_translation[0] + frame_translation[0],
            0,
            cumulative_translation[2] + frame_translation[2]
        ]
        
        # Apply cumulative translation to current frame
        guided_frames[i] = translate_pose(guided_frames[i], frame_translation)
    
    return guided_frames

def main(output_dir:str, lead_follow_height_ratio:float):
    ground_ankle_path = os.path.join(output_dir, 'all_floor_ankles.json')
    ground_ankles = utils.load_json_integer_keys(ground_ankle_path)

    lead_keypoints_path = os.path.join(output_dir, 'lead_smoothed_keypoints_3d.json')
    lead_keypoints = utils.load_json(lead_keypoints_path)

    follow_keypoints_path = os.path.join(output_dir, 'follow_smoothed_keypoints_3d.json')
    follow_keypoints = utils.load_json(follow_keypoints_path)

    camera_position_path = os.path.join(output_dir, 'initial_camera_pose.json')
    camera_position_data = utils.load_json(camera_position_path)
    camera_position = camera_position_data['position']

    # First scale the follow poses to match height ratio
    follow_keypoints = scale_poses_to_match_ratio(lead_keypoints, follow_keypoints, lead_follow_height_ratio)

    # Then process both dancers with rotation correction
    aligned_lead_keypoints = process_frame_discontinuities(
        lead_keypoints, ground_ankles, 'lead', camera_position)
    aligned_follow_keypoints = process_frame_discontinuities(
        follow_keypoints, ground_ankles, 'follow', camera_position)

    # Guide poses to ground positions with correct roles
    guided_lead_keypoints = guide_poses_to_ground(
        aligned_lead_keypoints,
        ground_ankles,
        'lead',
        camera_position)

    guided_follow_keypoints = guide_poses_to_ground(
        aligned_follow_keypoints,
        ground_ankles,
        'follow',
        camera_position)

    # Save final poses
    lead_output_path = os.path.join(output_dir, 'lead_aligned_keypoints_3d.json')
    follow_output_path = os.path.join(output_dir, 'follow_aligned_keypoints_3d.json')
    utils.save_json(guided_lead_keypoints, lead_output_path)
    utils.save_json(guided_follow_keypoints, follow_output_path)

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
