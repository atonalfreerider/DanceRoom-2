import os
import numpy as np
import json
from multiprocessing import Process, Queue
from tqdm import tqdm

import cv2
from scipy.spatial.transform import Rotation

def render(imagedir, calib, calibrated_camera, poses, stride=1, skip=0):
    first = True
    queue = Queue(maxsize=8)

    # Get the directory and base name from the input path
    output_dir = os.path.dirname(imagedir)
    base_name = os.path.splitext(os.path.basename(imagedir))[0]
    output_path = os.path.join(output_dir, f"{base_name}_debug_overlay.mp4")

    if os.path.isdir(imagedir):
        # Count frames for progress bar
        n_frames = len([f for f in os.listdir(imagedir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        # Get frame count from video
        cap = cv2.VideoCapture(imagedir)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    # Calculate actual number of frames after stride and skip
    n_frames = (n_frames - skip) // stride

    out = None
    initial_rotation = Rotation.from_quat(calibrated_camera['rotation'])
    count = 0

    # Initialize progress bar
    pbar = tqdm(total=n_frames, desc="Generating debug overlay video", unit="frames")

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if first:
            _, H, W = image.shape

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))
            first = False

        # Get the current camera pose from VO
        current_pose = poses[count]
        current_pose_rotation = Rotation.from_quat(current_pose[3:])

        # Calculate the current camera rotation
        current_camera_rotation = current_pose_rotation * initial_rotation

        # Draw origin corner overlay
        frame = image.permute(1, 2, 0).cpu().numpy().copy()
        camera_position = calibrated_camera['position']
        focal_length = calibrated_camera['focal_length']
        frame_with_overlay = draw_origin_corner(frame, camera_position, current_camera_rotation, focal_length,
                                              intrinsics.cpu().numpy())

        out.write(frame_with_overlay)
        count += 1
        pbar.update(1)

    pbar.close()
    reader.join()

    if out is not None:
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
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--camera_position', type=float, nargs=3, required=True, help='Calibrated camera position (x, y, z)')
    parser.add_argument('--camera_rotation', type=float, nargs=4, required=True, help='Calibrated camera rotation quaternion (x, y, z, w)')
    parser.add_argument('--focal_length', type=float, required=True, help='Focal length distance to the virtual image')

    args = parser.parse_args()

    # Create output JSON path by replacing .mp4 with _vo.json
    json_path = args.video_path.rsplit('.', 1)[0] + '_vo.json'
    result = load_json(json_path)

    calibrated_camera = {
        'position': np.array(args.camera_position),
        'rotation': np.array(args.camera_rotation),
        'focal_length': args.focal_length
    }

    poses = json_to_poses(result)
    render(args.video_path, args.calib, calibrated_camera, poses)
