import os
import numpy as np
import json
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm

def load_camera_frame0(video_path):
    """Load initial camera parameters from JSON if it exists"""
    json_path = os.path.splitext(video_path)[0] + '_camera_frame0.json'
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return {
                'position': np.array(data['position']),
                'rotation': np.array(data['rotation']),
                'focal_length': data['focal_length']
            }
    except FileNotFoundError:
        # Default camera parameters
        return {
            'position': np.array([0.0, 1.2, 3.5]),
            'rotation': np.array([0.0, 0.0, 0.0, 1.0]),  # Forward-facing quaternion
            'focal_length': 0.35
        }

def save_camera_frame0(video_path, camera_params):
    """Save camera parameters to JSON"""
    json_path = os.path.splitext(video_path)[0] + '_camera_frame0.json'
    data = {
        'position': camera_params['position'].tolist(),
        'rotation': camera_params['rotation'].tolist(),
        'focal_length': camera_params['focal_length']
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def calibrate_camera(video_path):
    """Interactive camera calibration function"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    H, W = frame.shape[:2]
    
    # Load or initialize camera parameters
    camera_params = load_camera_frame0(video_path)
    
    # Camera movement parameters
    pos_delta = 0.1  # Translation step size
    rot_delta = 0.02  # Rotation step size
    focal_delta = 0.01  # Focal length step size

    def log_camera_state():
        """Helper function to log camera state"""
        pos = camera_params['position']
        rot = camera_params['rotation']
        print(f"\nCamera State:")
        print(f"Position (XYZ): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"Rotation (XYZW): [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f}]")
        print(f"Focal Length: {camera_params['focal_length']:.3f}")

    calibration_done = False
    while not calibration_done:
        # Create copy of frame for drawing
        display_frame = frame.copy()
        
        # Calculate intrinsics based on frame size
        fx = fy = min(H, W)
        cx, cy = W/2, H/2
        intrinsics = np.array([fx, fy, cx, cy])

        # Draw origin overlay
        current_rotation = Rotation.from_quat(camera_params['rotation'])
        display_frame = draw_origin_corner(
            display_frame, 
            camera_params['position'],
            current_rotation,
            camera_params['focal_length'],
            intrinsics
        )

        # Show frame
        cv2.imshow('Camera Calibration', display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle keyboard input
        update = True
        if key == ord('w'):  # Forward
            camera_params['position'][2] -= pos_delta
        elif key == ord('s'):  # Backward
            camera_params['position'][2] += pos_delta
        elif key == ord('a'):  # Left
            camera_params['position'][0] -= pos_delta
        elif key == ord('d'):  # Right
            camera_params['position'][0] += pos_delta
        elif key == ord('q'):  # Up
            camera_params['position'][1] += pos_delta
        elif key == ord('e'):  # Down
            camera_params['position'][1] -= pos_delta
        elif key == ord('z'):  # Increase focal length
            camera_params['focal_length'] += focal_delta
        elif key == ord('x'):  # Decrease focal length
            camera_params['focal_length'] = max(0.1, camera_params['focal_length'] - focal_delta)
        elif key == 82:  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            camera_params['rotation'] = (rot * Rotation.from_quat(camera_params['rotation'])).as_quat()
        elif key == 84:  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            camera_params['rotation'] = (rot * Rotation.from_quat(camera_params['rotation'])).as_quat()
        elif key == 81:  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            camera_params['rotation'] = (rot * Rotation.from_quat(camera_params['rotation'])).as_quat()
        elif key == 83:  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            camera_params['rotation'] = (rot * Rotation.from_quat(camera_params['rotation'])).as_quat()
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            camera_params['rotation'] = (rot * Rotation.from_quat(camera_params['rotation'])).as_quat()
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            camera_params['rotation'] = (rot * Rotation.from_quat(camera_params['rotation'])).as_quat()
        elif key == 13:  # Enter key
            save_camera_frame0(video_path, camera_params)
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
    return camera_params, fps

def render(video_path, calibrated_camera, poses):
    """Render debug overlay video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    output_path = os.path.splitext(video_path)[0] + '_debug_overlay.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Calculate intrinsics
    fx = fy = min(H, W)
    cx, cy = W/2, H/2
    intrinsics = np.array([fx, fy, cx, cy])

    initial_rotation = Rotation.from_quat(calibrated_camera['rotation'])
    frame_idx = 0

    # Initialize progress bar
    pbar = tqdm(total=min(frame_count, len(poses)), desc="Rendering debug overlay")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get current camera pose
        if frame_idx < len(poses):
            current_pose = poses[frame_idx]
            current_pose_rotation = Rotation.from_quat(current_pose[3:])
            current_camera_rotation = current_pose_rotation * initial_rotation

            # Draw origin corner overlay
            frame_with_overlay = draw_origin_corner(
                frame.copy(),
                calibrated_camera['position'],
                current_camera_rotation,
                calibrated_camera['focal_length'],
                intrinsics
            )

            out.write(frame_with_overlay)
            frame_idx += 1
            pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"\nDebug overlay video saved to: {output_path}")

def reverse_project_point(world_point, position, rotation, focal_length):
    target = (world_point - position) / np.linalg.norm(world_point - position)
    return get_image_plane_coordinates(target, rotation, focal_length)

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

def draw_origin_corner(frame, camera_position, camera_rotation, focal_length, intrinsics):
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
        img_coords = reverse_project_point(point, camera_position, camera_rotation, focal_length)
        pixel_x = int(img_coords[0] * fx + cx)
        pixel_y = int(img_coords[1] * fy + cy)
        image_points.append((pixel_x, pixel_y))

    # Draw the origin corner
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z
    for i, color in enumerate(colors):
        cv2.line(frame, image_points[0], image_points[i+1], color, 2)

    return frame

def json_to_poses(data):
    """
    Convert pose data from JSON format back to the original poses numpy array.
    The poses array has format [x,y,z, qx,qy,qz,qw] for each pose.

    Args:
        data

    Returns:
        tuple: (poses array, timestamps array)
            - poses: numpy array of shape (N,7) containing [x,y,z, qx,qy,qz,qw]
            - tstamps: numpy array of timestamps
    """

    # Get number of poses
    n_poses = len(data)

    # Initialize arrays
    poses_per_frame = np.zeros((n_poses, 7))  # [x,y,z, qx,qy,qz,qw]

    # Iterate through the data and extract quaternion components
    for i, pose_data in enumerate(data):

        # Position data [x,y,z]
        poses_per_frame[i, 0] = pose_data[0]
        poses_per_frame[i, 1] = pose_data[1]
        poses_per_frame[i, 2] = pose_data[2]

        # Quaternion data [qx,qy,qz,qw]
        # Need to roll +1 to reverse the original roll(-1)
        poses_per_frame[i, 3] = pose_data[3]
        poses_per_frame[i, 4] = pose_data[4]
        poses_per_frame[i, 5] = pose_data[5]
        poses_per_frame[i, 6] = pose_data[6]

    return poses_per_frame

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    args = parser.parse_args()

    # First, run camera calibration
    calibrated_camera, fps = calibrate_camera(args.video_path)

    # Load VO poses if they exist
    json_path = args.video_path.rsplit('.', 1)[0] + '_vo.json'
    result = load_json(json_path)
    
    if result:
        poses = json_to_poses(result)
        render(args.video_path, calibrated_camera, poses)
