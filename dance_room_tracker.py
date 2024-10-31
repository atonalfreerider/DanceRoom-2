import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import utils
from cotracker_runner import CoTracker
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
        
        self.cotracker = CoTracker()
        self.virtualRoom = VirtualRoom()
        
        # Tracking state
        self.current_frame_idx = 0
        self.next_track_id = 0  # For generating unique IDs
        
        # Replace tracking_points with simple category mapping - remove floor
        self.point_categories = {
            'back_wall': set(),
            'left_wall': set(),
            'right_wall': set()
        }
        
        self.point_visibility = {}  # ID -> bool
        self.point_tracks = {}      # ID -> {track: array, visibility: array, start_frame: int}
        
        # Load pose detections
        self.pose_detections = utils.load_json(f"{output_dir}/detections.json")

        self.frame_height, self.frame_width = None, None
        self.current_frame = None
        
        self.mouse_data = {'clicked': False}

        self.track_display_frames = 50  # Number of frames to display in past/future
        
        # Store initial world positions of tracking points
        self.initial_world_positions = {}  # point_id -> (surface, world_pos)
        
        # Joint keypoint indices for tracking
        self.joint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  # All COCO joints

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
                self.find_new_points()
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
            print("Running initial point tracking...")
            success = self.track_points_chunk()  # Check if tracking succeeded
            if not success:
                print("Initial tracking failed!")
                return
            
            print("Saving pose assignments...")
            self.save_pose_assignments()
        
        print("Starting video loop...")
        
        # Step 4: Start tracking sequence
        if camera_oriented and poses_selected:
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
                
                if self.check_track_discontinuity():
                    print("Track discontinuity detected")
                    break
                
                self.current_frame = frame
                self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                
                # Check remaining track length and run new tracking if needed
                min_future_frames = float('inf')
                for track_data in self.point_tracks.values():
                    track_end = track_data['start_frame'] + len(track_data['track'])
                    remaining_frames = track_end - self.current_frame_idx
                    min_future_frames = min(min_future_frames, remaining_frames)
                
                print(f"\rFrame {self.current_frame_idx}, Future frames: {min_future_frames}", end='', flush=True)
                
                if min_future_frames < 10:
                    print(f"\nRunning new tracking at frame {self.current_frame_idx}")
                    if not self.track_points_chunk():
                        print("Tracking failed, stopping")
                        break
                
                self.update_camera_from_tracks()
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

    def track_points_chunk(self):
        """Track points for a 50-frame chunk"""
        MIN_WALL_POINTS = 20
        
        # Count current valid points
        valid_points = []
        point_mapping = []
        
        for point_id in self.point_categories['back_wall']:
            if point_id in self.point_tracks:
                track_data = self.point_tracks[point_id]
                frame_idx = self.current_frame_idx - track_data['start_frame']
                if 0 <= frame_idx < len(track_data['track']):
                    point = track_data['track'][frame_idx]
                    valid_points.append(point)
                    point_mapping.append(point_id)
        
        # If we don't have enough points, generate new ones
        if len(valid_points) < MIN_WALL_POINTS:
            print(f"\nNot enough valid points ({len(valid_points)}), generating new set")
            
            # Clear all existing points
            self.point_tracks.clear()
            self.point_categories['back_wall'].clear()
            self.point_visibility.clear()
            self.initial_world_positions.clear()
            
            # Generate new points
            self.find_new_points()
            
            # Get the new points for tracking
            valid_points = []
            point_mapping = []
            for point_id in self.point_categories['back_wall']:
                track_data = self.point_tracks[point_id]
                point = track_data['track'][0]  # New points only have initial position
                valid_points.append(point)
                point_mapping.append(point_id)
        
        if not valid_points:
            print("No points to track!")
            return False
        
        print(f"Tracking {len(valid_points)} points from frame {self.current_frame_idx}")
        
        # Track points
        pred_tracks, pred_visibility = self.cotracker.track(
            self.video_path, 
            valid_points,
            start_frame=self.current_frame_idx,
            num_frames=50
        )
        
        if pred_tracks is None:
            print("Tracking failed!")
            return False
        
        print(f"Got tracks of shape: {pred_tracks.shape}")
        
        # Update tracks for each point
        for idx, point_id in enumerate(point_mapping):
            point_track = np.array([frame_points[idx] for frame_points in pred_tracks])
            point_visibility = np.array([frame_vis[idx] for frame_vis in pred_visibility])
            
            # Create new track data
            self.point_tracks[point_id] = {
                'track': point_track,
                'visibility': point_visibility,
                'start_frame': self.current_frame_idx,
                'category': 'back_wall'
            }
            
            # Update visibility state based on current frame only
            self.point_visibility[point_id] = point_visibility[0]
        
        return True

    def find_new_points(self):
        """Find new points and categorize them with a grid sampling technique."""
        MIN_POINTS_PER_SURFACE = 30
        GRID_SIZE = 6  # Number of grid cells per row and column

        # Count current visible points per surface - remove floor
        surface_counts = {
            'back_wall': 0,
            'left_wall': 0,
            'right_wall': 0
        }

        for category in surface_counts.keys():
            surface_counts[category] = sum(1 for pid in self.point_categories[category]
                                           if self.point_visibility.get(pid, True))

        # Create mask excluding person regions and existing points
        person_mask = np.ones((self.frame_height, self.frame_width), dtype=np.uint8)

        # Mask out people and existing points
        poses = self.pose_detections[str(self.current_frame_idx)]
        padding = 5
        for pose in poses:
            bbox = pose['bbox']
            x_min = max(0, int(bbox[0]) - padding)
            x_max = min(self.frame_width, int(bbox[0] + bbox[2]) + padding)
            y_min = max(0, int(bbox[1]) - padding)
            y_max = min(self.frame_height, int(bbox[1] + bbox[3]) + padding)
            cv2.rectangle(person_mask, (x_min, y_min), (x_max, y_max), 0, -1)

        # Grid dimensions
        grid_height = self.frame_height // GRID_SIZE
        grid_width = self.frame_width // GRID_SIZE

        # Find new points for each category using a grid sampling technique
        for category in surface_counts:
            if surface_counts[category] < MIN_POINTS_PER_SURFACE:
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                masked_gray = cv2.multiply(gray, person_mask)

                remaining_points = MIN_POINTS_PER_SURFACE - surface_counts[category]
                points_per_grid = remaining_points // (GRID_SIZE * GRID_SIZE)

                # Iterate over each grid cell
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        # Define region of interest for this grid cell
                        x_start = j * grid_width
                        y_start = i * grid_height
                        x_end = min((j + 1) * grid_width, self.frame_width)
                        y_end = min((i + 1) * grid_height, self.frame_height)

                        # Masked region within the grid cell
                        cell_mask = person_mask[y_start:y_end, x_start:x_end]
                        cell_gray = masked_gray[y_start:y_end, x_start:x_end]

                        # Detect points within this grid cell
                        cell_points = cv2.goodFeaturesToTrack(
                            cell_gray,
                            maxCorners=points_per_grid,
                            qualityLevel=0.01,
                            minDistance=35,
                            blockSize=3,
                            useHarrisDetector=True,
                            k=0.04,
                            mask=cell_mask
                        )

                        if cell_points is not None:
                            cell_points = cell_points.reshape(-1, 2)
                            for point in cell_points:
                                # Convert cell point to image coordinates
                                point[0] += x_start
                                point[1] += y_start

                                # Project and categorize the point
                                intersection = self.virtualRoom.project_point_to_planes(point, self.camera_poses)
                                if intersection is not None:
                                    surface, world_pos = intersection[0], intersection[1]
                                    if surface == category:
                                        point_id = self.next_track_id
                                        self.next_track_id += 1

                                        # Add point to category
                                        self.point_categories[category].add(point_id)
                                        self.point_visibility[point_id] = True

                                        # Store initial world position
                                        self.initial_world_positions[point_id] = (surface, world_pos)

                                        # Initialize track
                                        track = np.tile(point, (50, 1))
                                        visibility = np.ones(50, dtype=bool)
                                        self.point_tracks[point_id] = {
                                            'track': track,
                                            'visibility': visibility,
                                            'start_frame': self.current_frame_idx,
                                            'category': category  # Add category to track data
                                        }

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

    def update_camera_from_tracks(self):
        """Update camera pose using geometric relationships and optimization"""

        def calculate_pan_adjustment(walls):
            """Calculate pan angle using x-shifts on back wall and z-shifts on side walls"""
            if not any(walls.values()):
                return 0
                    
            x_shifts = []  # From back wall
            z_shifts = []  # From side walls
            
            # Back wall x-shifts
            if walls['back_wall']:
                initial_points = np.array([p[0] for p in walls['back_wall']])
                current_points = np.array([p[1] for p in walls['back_wall']])
                x_diffs = current_points[:, 0] - initial_points[:, 0]  # X-axis differences
                # Scale by distance to back wall for correct angle
                camera_to_wall = abs(current_params['position'][2] + self.virtualRoom.room_depth/2)
                x_angles = np.arctan2(x_diffs, camera_to_wall)
                x_shifts.extend(x_angles)  # Store angles directly
            
            # Side walls z-shifts
            for wall in ['left_wall', 'right_wall']:
                if walls[wall]:
                    initial_points = np.array([p[0] for p in walls[wall]])
                    current_points = np.array([p[1] for p in walls[wall]])
                    z_diffs = current_points[:, 2] - initial_points[:, 2]  # Z-axis differences
                    # Scale by distance to side wall
                    if wall == 'left_wall':
                        camera_to_wall = abs(current_params['position'][0] + self.virtualRoom.room_width/2)
                        z_angles = np.arctan2(z_diffs, camera_to_wall)
                    else:  # right wall
                        camera_to_wall = abs(current_params['position'][0] - self.virtualRoom.room_width/2)
                        z_angles = np.arctan2(z_diffs, camera_to_wall)
                    z_shifts.extend(z_angles)  # Store angles directly
            
            # Combine angles using RANSAC-like approach
            all_angles = np.array(x_shifts + z_shifts)
            if len(all_angles) < 3:
                return 0
                    
            # Use median as robust estimator of the angle
            return np.median(all_angles)

        def calculate_tilt_adjustment(walls):
            """Calculate tilt angle using y-shifts on all walls"""
            y_angles = []
            
            for wall, wall_points in walls.items():
                if wall_points:
                    initial_points = np.array([p[0] for p in wall_points])
                    current_points = np.array([p[1] for p in wall_points])
                    y_diffs = current_points[:, 1] - initial_points[:, 1]  # Y-axis differences
                    
                    # Calculate distance to each wall for proper angle calculation
                    if wall == 'back_wall':
                        camera_to_wall = abs(current_params['position'][2] + self.virtualRoom.room_depth/2)
                    elif wall == 'left_wall':
                        camera_to_wall = abs(current_params['position'][0] + self.virtualRoom.room_width/2)
                    else:  # right wall
                        camera_to_wall = abs(current_params['position'][0] - self.virtualRoom.room_width/2)
                    
                    # Convert shifts to angles
                    angles = np.arctan2(y_diffs, camera_to_wall)
                    y_angles.extend(angles)
            
            if len(y_angles) < 3:
                return 0
                    
            # Use median as robust estimator of the angle
            return np.median(y_angles)

        def optimize_roll_and_zoom():
            """Optimize roll and zoom using iterative approach"""
            def calculate_error(params):
                test_rot = Rotation.from_euler('z', params[0])  # Roll angle
                test_focal = current_params['focal_length'] * (1 + params[1])  # Zoom factor
                
                total_error = 0
                num_points = 0
                
                # Project all initial world positions with test parameters
                for point_id, (initial_surface, initial_pos) in self.initial_world_positions.items():
                    if not self.point_visibility.get(point_id, False):
                        continue
                        
                    # Get current tracked point position
                    if point_id not in self.point_tracks:
                        continue
                        
                    track_data = self.point_tracks[point_id]
                    frame_idx = self.current_frame_idx - track_data['start_frame']
                    if not (0 <= frame_idx < len(track_data['track'])):
                        continue
                        
                    current_point = track_data['track'][frame_idx]
                    
                    # Project initial world position with test parameters
                    test_params = {
                        'position': current_params['position'],
                        'rotation': (test_rot * base_rotation).as_quat(),
                        'focal_length': test_focal
                    }
                    
                    # Project initial world position to image plane
                    rotation = Rotation.from_quat(test_params['rotation'])
                    point_cam = rotation.inv().apply(initial_pos - test_params['position'])
                    
                    if point_cam[2] <= 0:  # Behind camera
                        continue
                        
                    fx = fy = min(self.frame_height, self.frame_width)
                    cx, cy = self.frame_width/2, self.frame_height/2
                    x = -point_cam[0] / point_cam[2]
                    y = -point_cam[1] / point_cam[2]
                    proj_x = x * fx * test_focal + cx
                    proj_y = y * fy * test_focal + cy
                    
                    # Calculate error between projected initial position and current tracked position
                    dx = proj_x - current_point[0]
                    dy = proj_y - current_point[1]
                    total_error += dx*dx + dy*dy
                    num_points += 1
                
                if num_points == 0:
                    return float('inf')  # Return infinite error if no points were processed
                        
                return total_error / num_points

            # Grid search for roll and zoom
            best_error = float('inf')
            best_params = [0, 0]  # [roll_angle, zoom_factor]
            
            roll_range = np.linspace(-0.1, 0.1, 11)
            zoom_range = np.linspace(-0.1, 0.1, 11)
            
            base_rotation = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
            
            for roll in roll_range:
                for zoom in zoom_range:
                    error = calculate_error([roll, zoom])
                    if error < best_error:
                        best_error = error
                        best_params = [roll, zoom]
                
            print(f"Roll-Zoom optimization: best_error={best_error:.2f}, roll={best_params[0]:.4f}, zoom={best_params[1]:.4f}")
            return best_params

        # Initialize current frame values if not exist
        if self.current_frame_idx not in self.frame_rotations:
            prev_frame = max(k for k in self.frame_rotations.keys() if k < self.current_frame_idx)
            self.frame_rotations[self.current_frame_idx] = self.frame_rotations[prev_frame].copy()
            self.frame_focal_lengths[self.current_frame_idx] = self.frame_focal_lengths[prev_frame]

        current_params = self.get_current_camera_params()

        # Collect points for each wall - explicitly exclude pose points
        walls = {
            'back_wall': [],
            'left_wall': [],
            'right_wall': []
        }
        
        for point_id, (initial_surface, initial_pos) in self.initial_world_positions.items():
            if initial_surface not in walls:  # Skip floor and pose points
                continue
                
            if not self.point_visibility.get(point_id, False):
                continue
                
            # Get current point directly from track data
            if point_id in self.point_tracks:
                track_data = self.point_tracks[point_id]

                frame_idx = self.current_frame_idx - track_data['start_frame']
                
                if 0 <= frame_idx < len(track_data['track']):
                    current_point = track_data['track'][frame_idx]
                    
                    # Project current point to world
                    intersection = self.virtualRoom.project_point_to_planes(current_point, current_params)
                    if intersection is None or intersection[0] != initial_surface:
                        continue
                        
                    walls[initial_surface].append((initial_pos, intersection[1]))

        # Optimize roll and zoom together
        roll_zoom = optimize_roll_and_zoom()

        # Apply roll
        current_rot = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
        roll_rot = Rotation.from_euler('z', roll_zoom[0], degrees=False)
        self.frame_rotations[self.current_frame_idx] = (roll_rot * current_rot).as_quat()

        self.frame_focal_lengths[self.current_frame_idx] *= (1 + roll_zoom[1])

        # Perform geometric solving for pan and tilt
        pan_angle = calculate_pan_adjustment(walls)
        
        current_rot = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
        pan_rot = Rotation.from_euler('y', pan_angle, degrees=False)
        self.frame_rotations[self.current_frame_idx] = (pan_rot * current_rot).as_quat()

        tilt_angle = calculate_tilt_adjustment(walls)
        
        current_rot = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
        tilt_rot = Rotation.from_euler('x', -tilt_angle, degrees=False)
        self.frame_rotations[self.current_frame_idx] = (tilt_rot * current_rot).as_quat()

    def check_track_discontinuity(self):
        """Check if either lead or follow track is missing in current frame"""
        # Only check for pose discontinuity if we're beyond the current tracks
        min_future_frames = float('inf')
        for track_data in self.point_tracks.values():
            track_end = track_data['start_frame'] + len(track_data['track'])
            remaining_frames = track_end - self.current_frame_idx
            min_future_frames = min(min_future_frames, remaining_frames)
        
        # If we still have frames in our tracks, don't check for discontinuity
        if min_future_frames > 0:
            return False
        
        # Otherwise, check if poses are still visible
        current_poses = self.pose_detections.get(str(self.current_frame_idx), {})
        return any(pose['id'] == self.lead_track_id or pose['id'] == self.follow_track_id 
                  for pose in current_poses)

    def prepare_display_frame(self):
        """Prepare frame for display during tracking"""
        current_params = self.get_current_camera_params()
        
        display_frame = self.current_frame.copy()
        display_frame = self.virtualRoom.draw_virtual_room(display_frame, current_params)
        
        point_colors = {
            'back_wall': (0, 255, 0),    # Green
            'left_wall': (255, 0, 0),    # Blue
            'right_wall': (255, 255, 0), # Cyan
        }
        
        # Draw all tracks that include the current frame
        for point_id, track_data in self.point_tracks.items():
            track = track_data['track']
            visibility = track_data['visibility']
            start_frame = track_data['start_frame']
            category = track_data['category']
            color = point_colors[category]
            
            # Calculate frame index within this track
            track_frame_idx = self.current_frame_idx - start_frame
            
            # Only process if this track includes the current frame
            if 0 <= track_frame_idx < len(track):
                current_point = track[track_frame_idx]
                
                # Draw history (past frames)
                for i in range(max(0, track_frame_idx - self.track_display_frames), track_frame_idx):
                    if visibility[i]:
                        alpha = 0.5 * (1 - (track_frame_idx - i) / self.track_display_frames)
                        faded_color = tuple(int(c * alpha) for c in color)
                        pt1 = tuple(map(int, track[i]))
                        pt2 = tuple(map(int, track[i + 1]))
                        cv2.line(display_frame, pt1, pt2, faded_color, 1)
                
                # Draw future predictions
                for i in range(track_frame_idx, min(len(track) - 1, track_frame_idx + self.track_display_frames)):
                    if visibility[i]:
                        alpha = 0.5 * (1 - (i - track_frame_idx) / self.track_display_frames)
                        faded_color = tuple(int(c * alpha) for c in color)
                        pt1 = tuple(map(int, track[i]))
                        pt2 = tuple(map(int, track[i + 1]))
                        cv2.line(display_frame, pt1, pt2, faded_color, 1)
                
                # Draw current point
                pixel_pos = tuple(map(int, current_point))
                cv2.circle(display_frame, pixel_pos, 4, color, -1)  # Filled circle
                
                # If this is a room surface point, draw the initial surface projection
                if category in ['back_wall', 'left_wall', 'right_wall']:
                    if point_id in self.initial_world_positions:
                        initial_surface, initial_world_pos = self.initial_world_positions[point_id]
                        
                        # Project initial world position back using current frame's camera parameters
                        rotation = Rotation.from_quat(current_params['rotation'])
                        point_cam = rotation.inv().apply(initial_world_pos - current_params['position'])
                        
                        if point_cam[2] > 0:  # Only if point is in front of camera
                            fx = fy = min(self.frame_height, self.frame_width)
                            cx, cy = self.frame_width/2, self.frame_height/2
                            x = -point_cam[0] / point_cam[2]
                            y = -point_cam[1] / point_cam[2]
                            proj_x = int(x * fx * current_params['focal_length'] + cx)
                            proj_y = int(y * fy * current_params['focal_length'] + cy)
                            
                            # Draw open circle for initial surface projection
                            cv2.circle(display_frame, (proj_x, proj_y), 6, color, 1)
                            
                            # Draw line between current point and its target position
                            cv2.line(display_frame, pixel_pos, (proj_x, proj_y), color, 1)
                
                # Draw point ID
                cv2.putText(display_frame, f"{point_id}", 
                          (pixel_pos[0] + 5, pixel_pos[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add point counts in top-left corner
        y_offset = 30
        cv2.putText(display_frame, f"Frame: {self.current_frame_idx}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for category, points in self.point_categories.items():
            visible_count = sum(1 for pid in points if self.point_visibility.get(pid, False))
            color = point_colors[category]
            cv2.putText(display_frame, f"{category}: {visible_count}/{len(points)} points", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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

    def auto_track_sequence(self):
        """Automatically track through frames until a discontinuity is found"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        while True:
            ret, frame = cap.read()
            if not ret or self.check_track_discontinuity():
                break
                
            self.current_frame = frame
            self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            # Check remaining track length and run new tracking if needed
            min_future_frames = float('inf')
            for track_data in self.point_tracks.values():
                track_end = track_data['start_frame'] + len(track_data['track'])
                remaining_frames = track_end - self.current_frame_idx
                min_future_frames = min(min_future_frames, remaining_frames)
            
            print(f"Frame {self.current_frame_idx}, Minimum future frames: {min_future_frames}")
            
            if min_future_frames < 10:
                print(f"Running new tracking at frame {self.current_frame_idx}")
                self.track_points_chunk()
            
            self.update_camera_from_tracks()
            self.update_display(True, True)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_camera_tracking()
                self.save_pose_assignments()
            elif key == 27:  # ESC
                break
        
        cap.release()

    def should_add_new_points(self):
        """Check if we need to add new tracking points"""
        # Get minimum remaining future frames across all tracks
        min_future_frames = float('inf')
        current_time = self.current_frame_idx
        
        for track_data in self.point_tracks.values():
            track_end = track_data['start_frame'] + len(track_data['track'])
            remaining_frames = track_end - current_time
            min_future_frames = min(min_future_frames, remaining_frames)
        
        print(f"Minimum future frames: {min_future_frames}")
        return min_future_frames < 10
