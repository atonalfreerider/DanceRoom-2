import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import os
import json

import utils
from virtual_room import VirtualRoom


class DanceRoomTracker:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.initial_camera_pose_json_path = output_dir + '/initial_camera_pose.json'
        self.camera_tracking_json_path = output_dir + '/camera_tracking.json'
        self.pose_assignments_json_path = output_dir + '/pose_assignments.json'
        
        # Load initial camera pose (position and initial orientation/focal)
        self.initial_camera_pose = self.load_initial_camera_pose()
        
        # Per-frame camera tracking data
        self.frame_rotations = {}  # frame_idx -> quaternion
        self.frame_focal_lengths = {}  # frame_idx -> focal_length
        
        # Load existing tracking data if it exists
        tracking_data = utils.load_json(self.camera_tracking_json_path) or {}
        for frame_str, data in tracking_data.items():
            frame_idx = int(frame_str)
            self.frame_rotations[frame_idx] = np.array(data['rotation'])
            self.frame_focal_lengths[frame_idx] = data['focal_length']

        # Load pose assignments if exists
        pose_assignments = utils.load_json(self.pose_assignments_json_path) or {}
        self.lead_track_id = pose_assignments.get('lead')
        self.follow_track_id = pose_assignments.get('follow')
        
        self.virtualRoom = VirtualRoom()
        
        # Tracking state
        self.current_frame_idx = 0

        # Load pose detections and VO data at startup
        self.pose_detections = utils.load_json(f"{output_dir}/detections.json")
        self.vo_data = self.load_vo_data()
        
        # Add tracking for keypoints
        self.rotation_keypoints = {}  # frame_idx -> rotation quaternion
        self.focal_keypoints = {}     # frame_idx -> focal length
        
        # Always store frame 0 as first keypoint
        if self.initial_camera_pose:
            self.rotation_keypoints[0] = self.initial_camera_pose['rotation'].copy()
            self.focal_keypoints[0] = self.initial_camera_pose['focal_length']

        self.frame_height, self.frame_width = None, None
        self.current_frame = None
        
        self.mouse_data = {'clicked': False}

    def run_video_loop(self):
        """New initialization workflow"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        self.frame_height, self.frame_width = frame.shape[:2]
        self.current_frame = frame
        self.virtualRoom.set_frame(self.frame_height, self.frame_width)

        # Create window and set mouse callback
        cv2.namedWindow('Dance Room Tracker')
        cv2.setMouseCallback('Dance Room Tracker', self.mouse_callback)
        self.process_vo_data()

        self.update_display(True)
        
        playing = False
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.save_camera_tracking()
                self.save_pose_assignments()
                print("Saved tracking data")
            
            elif key == ord(' '):  # Toggle play/pause
                playing = not playing
                print(f"Playback: {'Playing' if playing else 'Paused'}")
            
            elif key == 83 or playing:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                self.current_frame = frame
                self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.update_display(not playing)  # Show controls when paused
            
            elif key == 81:
                # Move back one frame
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_pos = max(0, current_pos - 2)  # -2 because we'll read forward one frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret:
                    self.current_frame = frame
                    self.current_frame_idx = new_pos
                    self.update_display(not playing)
            
            elif not playing:  # Handle inputs when paused
                if self.handle_camera_movement(key):
                    # Camera was adjusted, store new keypoint and interpolate
                    if self.current_frame_idx > 0:  # Only store if not frame 0
                        self.rotation_keypoints[self.current_frame_idx] = self.frame_rotations[self.current_frame_idx].copy()
                        self.focal_keypoints[self.current_frame_idx] = self.frame_focal_lengths[self.current_frame_idx]
                        self.interpolate_between_keypoints(self.current_frame_idx)
                    self.update_display(True)
            
            elif key == 27:  # ESC
                print("\nExiting video loop...")
                break
        
        cap.release()
        cv2.destroyAllWindows()

    #region DRAW

    def update_display(self, paused):
        """Update display based on current state"""
        display_frame = self.current_frame.copy()

        # Get current camera parameters
        display_frame = self.virtualRoom.draw_virtual_room(
            display_frame,
            self.initial_camera_pose['position'],
            self.frame_rotations[self.current_frame_idx],
            self.frame_focal_lengths[self.current_frame_idx]
        )

        # Draw poses
        display_frame = self.draw_poses(display_frame,
                                        show_all=(paused and not (self.lead_track_id and self.follow_track_id)))

        # Add UI text
        if paused:
            if self.current_frame_idx == 0:
                cv2.putText(display_frame, "WASD: Move camera | QE: Up/Down | Arrows: Pan/Tilt | OP: Roll | ZX: Zoom",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Arrows: Pan/Tilt | OP: Roll | ZX: Zoom",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not (self.lead_track_id and self.follow_track_id):
                cv2.putText(display_frame, "Left click: Select lead | Right click: Select follow",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(display_frame, f"Frame: {self.current_frame_idx}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Dance Room Tracker', display_frame)

    def prepare_display_frame(self):
        """Prepare frame for display during tracking"""
        display_frame = self.current_frame.copy()
        display_frame = self.virtualRoom.draw_virtual_room(
            display_frame,
            self.initial_camera_pose['position'],
            self.frame_rotations[self.current_frame_idx],
            self.frame_focal_lengths[self.current_frame_idx]
        )

        # Add point counts in top-left corner
        y_offset = 30
        cv2.putText(display_frame, f"Frame: {self.current_frame_idx}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        return display_frame

    @staticmethod
    def draw_pose(image, keypoints, color, is_lead_or_follow=False):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]

        def is_valid_point(point):
            return point[0] != 0 or point[1] != 0

        for connection in connections:
            start_point = keypoints[connection[0]][:2]
            end_point = keypoints[connection[1]][:2]
            if is_valid_point(start_point) and is_valid_point(end_point):
                cv2.line(image, tuple(map(int, start_point)), tuple(map(int, end_point)), color, 2)

        for point in keypoints:
            if is_valid_point(point[:2]):
                cv2.circle(image, tuple(map(int, point[:2])), 3, color, -1)

        if is_lead_or_follow:
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

    def draw_poses(self, frame, show_all=True):
        """Draw all detected poses or only selected poses with track IDs"""
        poses = self.pose_detections[str(self.current_frame_idx)]
        
        for pose in poses:
            keypoints = np.array(pose['keypoints'])
            track_id = pose['id']
            
            # Skip if not showing all and this isn't a selected pose
            if not show_all and track_id not in [self.lead_track_id, self.follow_track_id]:
                continue
            
            # Determine color based on role
            if track_id == self.lead_track_id:
                color = (204, 102, 0)  # Green for lead
            elif track_id == self.follow_track_id:
                color = (127, 0, 255)  # Red for follow
            else:
                color = (128, 128, 128)  # Gray for unselected poses
            
            # Draw pose
            self.draw_pose(frame, keypoints, color, 
                          is_lead_or_follow=(track_id in [self.lead_track_id, self.follow_track_id]))
            
            # Draw track ID above head
            head_point = keypoints[0][:2]
            cv2.putText(frame, f"ID: {track_id}", 
                       (int(head_point[0]), int(head_point[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    #endregion

    #region POSE

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for pose selection and ray debug"""
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_data['clicked'] = True
            self.virtualRoom.mouse_callback(x, y,
                                            self.frame_height,
                                            self.frame_width,
                                            self.initial_camera_pose['position'],
                                            self.frame_rotations[self.current_frame_idx],
                                            self.frame_focal_lengths[self.current_frame_idx])

            # Original pose selection logic
            poses = self.pose_detections[str(self.current_frame_idx)]
            closest_track_id = self.find_closest_pose(x, y, poses)

            if closest_track_id:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.lead_track_id = closest_track_id
                else:
                    self.follow_track_id = closest_track_id

                # Update display immediately after pose selection
                self.update_display(False)

    @staticmethod
    def find_closest_pose(x, y, poses):
        """Find the track ID of the pose closest to clicked point"""
        min_dist = float('inf')
        closest_track_id = None
        
        for pose in poses:
            track_id = pose['id']
            keypoints = np.array(pose['keypoints'])
            # Use torso points to determine pose center
            torso_points = keypoints[[5,6,11,12]][:,:2]  # shoulders and hips
            center = np.mean(torso_points, axis=0)
            
            dist = np.sqrt((center[0] - x)**2 + (center[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_track_id = track_id
        
        return closest_track_id

    def check_track_discontinuity(self):
        # Otherwise, check if poses are still visible
        current_poses = self.pose_detections.get(str(self.current_frame_idx), {})
        return all(int(pose['id']) == self.lead_track_id or int(pose['id']) == self.follow_track_id
                  for pose in current_poses)

    #endregion

    #region CAMERA

    def handle_camera_movement(self, key):
        """Handle keyboard input for camera movement"""
        if self.current_frame_idx == 0:
            # Allow full camera control at frame 0
            return self._handle_full_camera_movement(key)
        else:
            # Only allow rotation and focal length changes at other frames
            return self._handle_orientation_only(key)

    def _handle_full_camera_movement(self, key):
        """Handle only rotation and focal length changes"""
        rot_delta = 0.005
        focal_delta = 0.01
        current_rotation = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
        current_focal_length = self.frame_focal_lengths[self.current_frame_idx]

        update = False
        if key == ord('z'):  # Increase focal length
            self.frame_focal_lengths[self.current_frame_idx] = current_focal_length + focal_delta
            self.initial_camera_pose['focal_length'] = self.frame_focal_lengths[self.current_frame_idx]
            update = True
        elif key == ord('x'):  # Decrease focal length
            self.frame_focal_lengths[self.current_frame_idx] = max(0.1, current_focal_length - focal_delta)
            self.initial_camera_pose['focal_length'] = self.frame_focal_lengths[self.current_frame_idx]
            update = True
        elif key == ord('i'):  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
            update = True
        elif key == ord('k'):  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
            update = True
        elif key == ord('j'):  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
            update = True
        elif key == ord('l'):  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
            update = True
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
            update = True
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
            update = True

        if update:
            self.save_initial_camera_pose()

        return update

    def _handle_orientation_only(self, key):
        """Handle only rotation and focal length changes"""
        rot_delta = 0.005
        focal_delta = 0.01
        current_rotation = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
        current_focal_length = self.frame_focal_lengths[self.current_frame_idx]

        update = False
        if key == ord('z'):  # Increase focal length
            self.frame_focal_lengths[self.current_frame_idx] = current_focal_length + focal_delta
            update = True
        elif key == ord('x'):  # Decrease focal length
            self.frame_focal_lengths[self.current_frame_idx] = max(0.1, current_focal_length - focal_delta)
            update = True
        elif key == ord('i'):  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            update = True
        elif key == ord('k'):  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            update = True
        elif key == ord('j'):  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            update = True
        elif key == ord('l'):  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            update = True
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            update = True
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            update = True

        return update

    @staticmethod
    def slerp(t, q0, q1):
        """
        Perform spherical linear interpolation between two quaternions.

        Parameters:
        - t: Interpolation factor, should be between 0 and 1.
        - q0, q1: scipy Rotation objects representing the start and end orientations.

        Returns:
        - Interpolated quaternion as a numpy array [w, x, y, z].
        """
        # Convert Rotation objects to quaternions
        q0 = q0.as_quat()
        q1 = q1.as_quat()

        # Normalize the quaternions to be sure
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)

        # Compute the dot product (cosine of the angle between them)
        dot_product = np.dot(q0, q1)

        # If the dot product is negative, invert one of the quaternions
        # to take the shortest path
        if dot_product < 0.0:
            q1 = -q1
            dot_product = -dot_product

        # Clamp dot_product to avoid numerical errors leading to values outside [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate the angle between the quaternions
        theta_0 = np.arccos(dot_product)  # Initial angle
        theta = theta_0 * t  # Interpolated angle

        # Compute the two quaternion scales
        sin_theta_0 = np.sin(theta_0)
        if sin_theta_0 < 1e-6:
            # If the angle is small, use linear interpolation to avoid division by zero
            q_interp = (1 - t) * q0 + t * q1
        else:
            sin_theta = np.sin(theta)
            scale_0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
            scale_1 = sin_theta / sin_theta_0
            q_interp = scale_0 * q0 + scale_1 * q1

        # Normalize the result
        q_interp = q_interp / np.linalg.norm(q_interp)

        # Return as a Rotation object
        return Rotation.from_quat(q_interp)

    def interpolate_between_keypoints(self, new_keyframe_idx):
        """Interpolate camera parameters when a new keypoint is added"""
        if new_keyframe_idx == 0:
            return  # No interpolation needed for frame 0

        keyframes = sorted(self.rotation_keypoints.keys())
        new_kf_index = keyframes.index(new_keyframe_idx)
        
        # Get previous keyframe
        prev_keyframe = keyframes[new_kf_index - 1]
        
        # Get next keyframe if it exists
        next_keyframe = None
        if new_kf_index < len(keyframes) - 1:
            next_keyframe = keyframes[new_kf_index + 1]
        
        # Interpolate between previous and new keyframe
        self.interpolate_range(prev_keyframe, new_keyframe_idx)
        
        # If there's a next keyframe, interpolate between new and next
        if next_keyframe is not None:
            self.interpolate_range(new_keyframe_idx, next_keyframe)
        else:
            # Interpolate from new keyframe to end of video
            self.interpolate_to_end(new_keyframe_idx)

    def interpolate_range(self, start_frame, end_frame):
        """Interpolate camera parameters between two keyframes"""
        start_rot = Rotation.from_quat(self.rotation_keypoints[start_frame])
        end_rot = Rotation.from_quat(self.rotation_keypoints[end_frame])
        start_focal = self.focal_keypoints[start_frame]
        end_focal = self.focal_keypoints[end_frame]
        
        for frame in range(start_frame + 1, end_frame):
            t = (frame - start_frame) / (end_frame - start_frame)
            
            # Interpolate rotation
            interpolated_rot = self.slerp(t, start_rot, end_rot)
            self.frame_rotations[frame] = interpolated_rot.as_quat()
            
            # Interpolate focal length
            self.frame_focal_lengths[frame] = start_focal * (1 - t) + end_focal * t

    def interpolate_to_end(self, start_frame):
        """Interpolate camera parameters from last keyframe to end of video"""
        if not self.vo_data:
            return
            
        start_rot = Rotation.from_quat(self.rotation_keypoints[start_frame])
        start_focal = self.focal_keypoints[start_frame]
        
        # Use VO data to determine end of video
        for frame in range(start_frame + 1, len(self.vo_data)):
            self.frame_rotations[frame] = start_rot.as_quat()
            self.frame_focal_lengths[frame] = start_focal

    #endregion

    # region FILE IO

    def load_initial_camera_pose(self):
        """Load initial camera pose from JSON if it exists"""
        data = utils.load_json(self.initial_camera_pose_json_path)
        if data:
            return {
                'position': np.array(data['position']),
                'rotation': np.array(data['rotation']),
                'focal_length': data['focal_length']
            }
        else:
            # Default camera pose: at +Z looking towards -Z (back wall)
            # Camera height is now positive since y=0 is at floor level
            return {
                'position': np.array([0.0, 1.1, 3.4]),  # Position with positive Y for height above floor
                'rotation': np.array([0.0, 1.0, 0.0, 0.0]),  # Looking towards -Z
                'focal_length': 1.400
            }

    def save_initial_camera_pose(self):
        """Save camera poses to JSON"""
        data = {
            'position': self.initial_camera_pose['position'].tolist(),
            'rotation': self.initial_camera_pose['rotation'].tolist(),
            'focal_length': self.initial_camera_pose['focal_length']
        }
        utils.save_json(data, self.initial_camera_pose_json_path)
        print(f'saved to {self.initial_camera_pose_json_path}')

    def save_camera_tracking(self):
        """Save per-frame camera tracking data"""
        tracking_data = {}
        for frame_idx in self.frame_rotations.keys():
            tracking_data[str(frame_idx)] = {
                'rotation': self.frame_rotations[frame_idx].tolist(),
                'focal_length': self.frame_focal_lengths[frame_idx]
            }
        utils.save_json(tracking_data, self.camera_tracking_json_path)
        print(f'Saved camera tracking to {self.camera_tracking_json_path}')

    def save_pose_assignments(self):
        """Save lead/follow pose assignments"""
        assignments = {
            'lead': self.lead_track_id,
            'follow': self.follow_track_id
        }
        utils.save_json(assignments, self.pose_assignments_json_path)
        print(f'Saved pose assignments to {self.pose_assignments_json_path}')

    def load_vo_data(self):
        """Load visual odometry data from json"""
        # Construct VO json path
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        vo_path = os.path.join(self.output_dir, f"{video_name}_vo.json")

        try:
            with open(vo_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: No VO data found at {vo_path}")
            return None

    def process_vo_data(self):
        """Process visual odometry data after camera calibration"""
        vo_data = self.load_vo_data()
        if not vo_data:
            return

        # Get initial camera state
        initial_rotation = Rotation.from_quat(self.initial_camera_pose['rotation'])
        initial_focal = self.initial_camera_pose['focal_length']
        initial_z = vo_data[0][2]  # Initial z position from VO

        # Process each frame
        window_size = 20
        raw_rotations = []  # Store raw rotation matrices for smoothing

        # First pass: calculate raw rotations
        for frame_data in vo_data:
            vo_rotation = Rotation.from_quat(frame_data[3:])
            relative_rotation = initial_rotation * vo_rotation.inv()
            raw_rotations.append(relative_rotation.as_matrix())

        # Apply moving average smoothing to rotation matrices
        smoothed_rotations = []
        for i in range(len(raw_rotations)):
            # Calculate window bounds
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(raw_rotations), i + window_size // 2)

            # Get window of rotation matrices
            window = raw_rotations[start_idx:end_idx]

            # Average the rotation matrices
            avg_matrix = np.mean(window, axis=0)

            # Project back to valid rotation matrix using SVD
            u, _, vh = np.linalg.svd(avg_matrix)
            smoothed_matrix = u @ vh
            smoothed_rotations.append(Rotation.from_matrix(smoothed_matrix))

        # Second pass: store smoothed rotations and process focal lengths
        for frame_idx, frame_data in enumerate(vo_data):
            # Store smoothed rotation
            self.frame_rotations[frame_idx] = smoothed_rotations[frame_idx].as_quat()

            # Process focal length as before
            pos = np.array(frame_data[:3])
            z_delta = pos[2] - initial_z
            focal_delta = z_delta * 0.2 # RATIO of Z to Focal Length
            self.frame_focal_lengths[frame_idx] = initial_focal + focal_delta

        # Save updated tracking data
        self.save_camera_tracking()

    # endregion