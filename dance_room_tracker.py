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
        self.camera_poses_json_path = output_dir + '/camera_poses.json'
        self.camera_tracking_json_path = output_dir + '/camera_tracking.json'
        self.pose_assignments_json_path = output_dir + '/pose_assignments.json'
        
        # Load initial camera pose (position and initial orientation/focal)
        self.camera_poses = self.load_camera_poses()
        
        # Per-frame camera tracking data
        self.frame_rotations = {}  # frame_idx -> quaternion
        self.frame_focal_lengths = {}  # frame_idx -> focal_length
        
        # Load existing tracking data if it exists
        tracking_data = utils.load_json(self.camera_tracking_json_path) or {}
        for frame_str, data in tracking_data.items():
            frame_idx = int(frame_str)
            self.frame_rotations[frame_idx] = np.array(data['rotation'])
            self.frame_focal_lengths[frame_idx] = data['focal_length']
        
        # Load camera tracking data if exists
        self.camera_tracking = utils.load_json(self.camera_tracking_json_path) or {}
        
        # Load pose assignments if exists
        pose_assignments = utils.load_json(self.pose_assignments_json_path) or {}
        self.lead_track_id = pose_assignments.get('lead')
        self.follow_track_id = pose_assignments.get('follow')
        
        self.virtualRoom = VirtualRoom()
        
        # Tracking state
        self.current_frame_idx = 0

        # Load pose detections
        self.pose_detections = utils.load_json(f"{output_dir}/detections.json")

        self.frame_height, self.frame_width = None, None
        self.current_frame = None
        
        self.mouse_data = {'clicked': False}

    def initialize_tracking(self):
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
        
        # State flags
        camera_oriented = False
        poses_selected = self.lead_track_id is not None and self.follow_track_id is not None
        
        print("\nInitializing tracking...")
        print(f"Camera oriented: {camera_oriented}")
        print(f"Poses selected: {poses_selected}")
        
        # Initial display
        self.update_display(camera_oriented, poses_selected)
        
        # Step 1: Camera Orientation
        while not camera_oriented:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                camera_oriented = True
                self.save_camera_poses()
                # Store initial rotation and focal length for frame 0
                self.frame_rotations[0] = self.camera_poses['rotation'].copy()
                self.frame_focal_lengths[0] = self.camera_poses['focal_length']
                print("Finding initial tracking points...")
                self.update_display(camera_oriented, poses_selected)
            elif key == 27:  # ESC - return to camera adjustment
                camera_oriented = False
            elif key != 255:  # Only handle other keys if they're actually pressed
                if self.handle_camera_movement(key):
                    self.update_display(camera_oriented, poses_selected)
        
        print("Camera oriented, proceeding to pose selection...")

        # Step 2: Pose Selection (if not already selected)
        if not poses_selected:
            self.update_display(True, False)
            while not poses_selected:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('0'):
                    self.lead_track_id = None
                    self.follow_track_id = None
                    self.update_display(True, False)
                elif key == 27:  # ESC - return to camera orientation
                    camera_oriented = False
                    self.update_display(camera_oriented, poses_selected)
                    break
                elif self.mouse_data.get('clicked', True):
                    # Both poses are now selected
                    self.mouse_data['clicked'] = False
                    poses_selected = True

        # Step 3: Initialize tracking (whether poses were just selected or pre-selected)
        if poses_selected:
            print("Saving pose assignments...")
            self.save_pose_assignments()
        
        print("Starting video loop...")
        
        # Step 4: Start tracking sequence
        if camera_oriented and poses_selected:
            self.process_vo_data()
            self.run_video_loop()
        else:
            print("Restarting initialization due to incomplete setup...")
            self.initialize_tracking()

    def update_display(self, camera_oriented, poses_selected):
        """Update display based on current state"""
        display_frame = self.current_frame.copy()
        
        if not camera_oriented:
            display_frame = self.virtualRoom.draw_virtual_room(display_frame, self.camera_poses)
            cv2.putText(display_frame, "Camera Orientation Mode - Press ENTER when done", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "WASD: Move camera | QE: Up/Down", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Arrows: Pan/Tilt | OP: Roll | ZX: Zoom", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        elif not poses_selected:
            # Draw tracking points first
            display_frame = self.prepare_display_frame()
            # Then draw all poses on top
            display_frame = self.draw_poses(display_frame, show_all=True)
            cv2.putText(display_frame, "Pose Selection Mode", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Left click: Select lead | Right click: Select follow", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 0 to reset selection", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:
            # Draw tracking points first
            display_frame = self.prepare_display_frame()
            # Then draw only selected poses
            display_frame = self.draw_poses(display_frame, show_all=False)
            cv2.putText(display_frame, "SPACE: run to next discontinuity | RIGHT: step one frame",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Dance Room Tracker', display_frame)

    def run_video_loop(self):
        """Main video loop"""
        print("Entering video loop...")
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        auto_play = False
        
        while True:
            if auto_play:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.save_camera_tracking()
                self.save_pose_assignments()
                print("Saved tracking data")
            
            elif key == ord(' '):  # Toggle auto-play
                auto_play = not auto_play
                print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
            
            elif key == 83 or auto_play:  # Right arrow or auto-play
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                self.current_frame = frame
                self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                if self.check_track_discontinuity():
                    print("Track discontinuity detected")
                    break

                self.update_display(True, True)
            
            elif key == 27:  # ESC
                print("\nExiting video loop...")
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for pose selection and ray debug"""
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_data['clicked'] = True
            self.virtualRoom.mouse_callback(x, y, self.frame_height, self.frame_width, self.camera_poses)

            # Original pose selection logic
            poses = self.pose_detections[str(self.current_frame_idx)]
            closest_track_id = self.find_closest_pose(x, y, poses)
            
            if closest_track_id:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.lead_track_id = closest_track_id
                else:
                    self.follow_track_id = closest_track_id
                
                # Update display immediately after pose selection
                self.update_display(True, False)

    def load_camera_poses(self):
        """Load initial camera poses from JSON if it exists"""
        data = utils.load_json(self.camera_poses_json_path)
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

    def save_camera_poses(self):
        """Save camera poses to JSON"""
        data = {
            'position': self.camera_poses['position'].tolist(),
            'rotation': self.camera_poses['rotation'].tolist(),
            'focal_length': self.camera_poses['focal_length']
        }
        utils.save_json(data, self.camera_poses_json_path)
        print(f'saved to {self.camera_poses_json_path}')

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

    def handle_camera_movement(self, key):
        """Handle keyboard input for camera movement"""
        pos_delta = 0.1
        rot_delta = 0.005
        focal_delta = 0.01
        rotation = Rotation.from_quat(self.camera_poses['rotation'])

        update = False
        if key == ord('w'):  # Forward
            self.camera_poses['position'][2] -= pos_delta
            update = True
        elif key == ord('s'):  # Backward
            self.camera_poses['position'][2] += pos_delta
            update = True
        elif key == ord('a'):  # Left
            self.camera_poses['position'][0] -= pos_delta
            update = True
        elif key == ord('d'):  # Right
            self.camera_poses['position'][0] += pos_delta
            update = True
        elif key == ord('q'):  # Up
            self.camera_poses['position'][1] += pos_delta
            update = True
        elif key == ord('e'):  # Down
            self.camera_poses['position'][1] -= pos_delta
            update = True
        elif key == ord('z'):  # Increase focal length
            self.camera_poses['focal_length'] += focal_delta
            update = True
        elif key == ord('x'):  # Decrease focal length
            self.camera_poses['focal_length'] = max(0.1, self.camera_poses['focal_length'] - focal_delta)
            update = True
        elif key == 82:  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
            update = True
        elif key == 84:  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
            update = True
        elif key == 81:  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
            update = True
        elif key == 83:  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
            update = True
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
            update = True
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
            update = True

        if update:
            # Log camera state
            pos = self.camera_poses['position']
            rot = self.camera_poses['rotation']
            print(f"\nCamera State:")
            print(f"Position (XYZ): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"Rotation (XYZW): [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f}]")
            print(f"Focal Length: {self.camera_poses['focal_length']:.3f}")

            # Update display immediately
            self.update_display(False, False)

        return update

    def check_track_discontinuity(self):
        # Otherwise, check if poses are still visible
        current_poses = self.pose_detections.get(str(self.current_frame_idx), {})
        return all(int(pose['id']) == self.lead_track_id or int(pose['id']) == self.follow_track_id
                  for pose in current_poses)

    def prepare_display_frame(self):
        """Prepare frame for display during tracking"""
        current_params = self.get_current_camera_params()
        
        display_frame = self.current_frame.copy()
        display_frame = self.virtualRoom.draw_virtual_room(display_frame, current_params)
        
        # Add point counts in top-left corner
        y_offset = 30
        cv2.putText(display_frame, f"Frame: {self.current_frame_idx}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        return display_frame

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

    def get_current_camera_params(self):
        """Get camera parameters for current frame"""
        # Use frame 0 values if current frame not yet set
        rotation = self.frame_rotations.get(
            self.current_frame_idx, 
            self.frame_rotations.get(0, self.camera_poses['rotation'])
        )
        focal_length = self.frame_focal_lengths.get(
            self.current_frame_idx, 
            self.frame_focal_lengths.get(0, self.camera_poses['focal_length'])
        )
        return {
            'position': self.camera_poses['position'],  # Fixed position
            'rotation': rotation,
            'focal_length': focal_length
        }

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
        initial_rotation = Rotation.from_quat(self.camera_poses['rotation'])
        initial_focal = self.camera_poses['focal_length']
        initial_z = vo_data[0][2]  # Initial z position from VO

        # Process each frame
        for frame_idx, frame_data in enumerate(vo_data):
            # Extract position and rotation from VO data
            pos = np.array(frame_data[:3])
            vo_rotation = Rotation.from_quat(frame_data[3:])

            # Transform rotation relative to initial camera rotation
            # Invert the VO rotation to correct the reversed motions
            relative_rotation = initial_rotation * vo_rotation.inv()
            self.frame_rotations[frame_idx] = relative_rotation.as_quat()

            # Convert z-translation to focal length change (10:1 ratio)
            z_delta = pos[2] - initial_z
            focal_delta = z_delta * 0.1
            self.frame_focal_lengths[frame_idx] = initial_focal + focal_delta

        # Save updated tracking data
        self.save_camera_tracking()
