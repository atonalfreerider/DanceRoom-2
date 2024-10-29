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
        self.camera_poses = self.load_camera_poses()
        self.cotracker = CoTracker()
        self.virtualRoom = VirtualRoom()
        
        # Tracking state
        self.lead_track_id = None
        self.follow_track_id = None
        self.current_frame_idx = 0
        self.tracking_points = {
            'floor': [],
            'back_wall': [],
            'left_wall': [],
            'right_wall': [],
            'lead_extremities': [],
            'follow_extremities': []
        }
        
        # Load pose detections
        self.pose_detections = utils.load_json(f"{output_dir}/detections.json")

        self.frame_height, self.frame_width = None, None
        self.current_frame = None
        
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
        
        # Main initialization loop
        while True:
            display_frame = self.current_frame.copy()
            
            # Draw virtual room
            display_frame = self.virtualRoom.draw_virtual_room(display_frame, self.camera_poses)
            
            # Draw poses with track IDs
            display_frame = self.draw_poses(display_frame)
            
            cv2.imshow('Dance Room Tracker', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and self.lead_track_id and self.follow_track_id:
                # Initialize tracking points
                self.initialize_tracking_points()
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            
            # Handle camera movement keys
            self.handle_camera_movement(key)
        
        cap.release()
        cv2.destroyAllWindows()

    def run_tracking(self):
        """Main tracking loop"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            self.current_frame = frame
            self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Track points in 50-frame chunks
            if self.current_frame_idx % 50 - 1 == 0:
                self.track_points_chunk()
            
            # Check for pose track discontinuity
            if self.check_track_discontinuity():
                display_frame = self.prepare_display_frame()
                cv2.imshow('Dance Room Tracker', display_frame)
                
                # Wait for user to select new track or navigate
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('n'):  # Continue tracking
                        break
                    elif key == 81:  # Left arrow - review previous frame
                        self.current_frame_idx = max(0, self.current_frame_idx - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                    elif key == 83:  # Right arrow - next frame
                        break
                    elif key == 27:  # ESC
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            
        cap.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for pose selection and ray debug"""
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.virtualRoom.mouse_callback(x, y, self.frame_height, self.frame_width, self.camera_poses)

            # Original pose selection logic
            poses = self.pose_detections[str(self.current_frame_idx)]
            closest_track_id = self.find_closest_pose(x, y, poses)
            
            if closest_track_id:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.lead_track_id = closest_track_id
                else:
                    self.follow_track_id = closest_track_id

    def initialize_tracking_points(self):
        """Find and initialize tracking points in the scene"""
        # Get current frame poses and create mask for person regions
        poses = self.pose_detections[str(self.current_frame_idx)]
        person_mask = np.ones((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # Create mask excluding person regions (with padding)
        padding = 5
        for pose in poses:
            # Use bbox directly from pose data
            bbox = pose['bbox']  # [x, y, width, height]
            x_min = max(0, int(bbox[0]) - padding)
            x_max = min(self.frame_width, int(bbox[0] + bbox[2]) + padding)
            y_min = max(0, int(bbox[1]) - padding)
            y_max = min(self.frame_height, int(bbox[1] + bbox[3]) + padding)
            
            # Black out the rectangular region
            cv2.rectangle(person_mask, (x_min, y_min), (x_max, y_max), 0, -1)

        # Convert frame to grayscale
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to grayscale image
        masked_gray = cv2.multiply(gray, person_mask)
        
        # Optional: Display mask for debugging
        cv2.imshow('Person Mask', person_mask * 255)

        # Find good features to track with adjusted parameters
        points = cv2.goodFeaturesToTrack(
            masked_gray,
            maxCorners=2000,
            qualityLevel=0.005,
            minDistance=2,
            blockSize=3,
            useHarrisDetector=True,
            k=0.04,
            mask=person_mask
        )

        # Clear previous points
        self.tracking_points = {
            'floor': [],
            'back_wall': [],
            'left_wall': [],
            'right_wall': [],
            'lead_extremities': [],
            'follow_extremities': []
        }

        if points is not None:
            points = points.reshape(-1, 2)
            
            # Project each point and categorize it
            for point in points:
                world_point = self.virtualRoom.project_point_to_planes(point, self.camera_poses)
                if world_point is not None:
                    self.virtualRoom.categorize_tracking_point(point, world_point, self.tracking_points)

            # Debug print
            print("Found tracking points:")
            for category, points in self.tracking_points.items():
                print(f"{category}: {len(points)} points")

        # Initialize pose extremity points
        self.initialize_pose_extremity_points()

    def track_points_chunk(self):
        """Track points for a 50-frame chunk"""
        # Prepare points for cotracker
        all_points = []
        all_points.extend(self.tracking_points['floor'])
        all_points.extend(self.tracking_points['back_wall'])
        all_points.extend(self.tracking_points['left_wall'])
        all_points.extend(self.tracking_points['right_wall'])
        all_points.extend(self.tracking_points['lead_extremities'])
        all_points.extend(self.tracking_points['follow_extremities'])
        
        # Track points
        pred_tracks, pred_visibility = self.cotracker.track(
            self.video_path, 
            all_points,
            start_frame=self.current_frame_idx,
            num_frames=50
        )
        
        # Update camera pose based on wall and floor points
        self.update_camera_from_tracks(pred_tracks, pred_visibility)

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

    def render(self):
        """Render debug overlay video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

    def reverse_project_point(self, world_point, position, rotation, focal_length):
        target = (world_point - position) / np.linalg.norm(world_point - position)
        return self.get_image_plane_coordinates(target, rotation, focal_length)

    @staticmethod
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

    def draw_origin_corner(self, frame, camera_position, camera_rotation, focal_length, intrinsics):
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
            img_coords = self.reverse_project_point(point, camera_position, camera_rotation, focal_length)
            pixel_x = int(img_coords[0] * fx + cx)
            pixel_y = int(img_coords[1] * fy + cy)
            image_points.append((pixel_x, pixel_y))

        # Draw the origin corner
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue for X, Y, Z
        for i, color in enumerate(colors):
            cv2.line(frame, image_points[0], image_points[i+1], color, 2)

        return frame

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

    def draw_poses(self, frame):
        """Draw all detected poses with track IDs"""
        poses = self.pose_detections[str(self.current_frame_idx)]
        
        for pose in poses:
            keypoints = np.array(pose['keypoints'])
            track_id = pose['id']
            
            # Determine color based on role
            if track_id == self.lead_track_id:
                color = (0, 255, 0)  # Green for lead
            elif track_id == self.follow_track_id:
                color = (0, 0, 255)  # Red for follow
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

        def log_camera_state():
            """Helper function to log camera state"""
            pos = self.camera_poses['position']
            rot = self.camera_poses['rotation']
            print(f"\nCamera State:")
            print(f"Position (XYZ): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print(f"Rotation (XYZW): [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f}]")
            print(f"Focal Length: {self.camera_poses['focal_length']:.3f}")

        pos_delta = 0.1
        rot_delta = 0.005
        focal_delta = 0.01
        rotation = Rotation.from_quat(self.camera_poses['rotation'])

        update = True
        if key == ord('w'):  # Forward
            self.camera_poses['position'][2] -= pos_delta
        elif key == ord('s'):  # Backward
            self.camera_poses['position'][2] += pos_delta
        elif key == ord('a'):  # Left
            self.camera_poses['position'][0] -= pos_delta
        elif key == ord('d'):  # Right
            self.camera_poses['position'][0] += pos_delta
        elif key == ord('q'):  # Up
            self.camera_poses['position'][1] += pos_delta
        elif key == ord('e'):  # Down
            self.camera_poses['position'][1] -= pos_delta
        elif key == ord('z'):  # Increase focal length
            self.camera_poses['focal_length'] += focal_delta
        elif key == ord('x'):  # Decrease focal length
            self.camera_poses['focal_length'] = max(0.1, self.camera_poses['focal_length'] - focal_delta)
        elif key == 82:  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
        elif key == 84:  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
        elif key == 81:  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
        elif key == 83:  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.camera_poses['rotation'] = (rot * rotation).as_quat()
        elif key == 13: # Enter (save)
            self.save_camera_poses()
            update = False
        else:
            update = False

        # Log camera state if there was an update
        if update:
            log_camera_state()

    def initialize_pose_extremity_points(self):
        """Initialize tracking points for lead and follow pose extremities"""
        # Get poses for lead and follow
        poses = self.pose_detections[str(self.current_frame_idx)]
        
        # Find the pose data for lead and follow by ID
        lead_pose = None
        follow_pose = None
        for pose in poses:
            if pose['id'] == self.lead_track_id:
                lead_pose = np.array(pose['keypoints'])
            elif pose['id'] == self.follow_track_id:
                follow_pose = np.array(pose['keypoints'])
        
        if lead_pose is None or follow_pose is None:
            print(f"Warning: Could not find lead ({self.lead_track_id}) or follow ({self.follow_track_id}) pose")
            return
        
        # Define extremity indices (hands, elbows, feet, knees)
        extremity_indices = [
            7, 8, 9, 10,  # hands and elbows
            13, 14, 15, 16  # knees and feet
        ]
        
        # Add lead extremities
        for idx in extremity_indices:
            point = lead_pose[idx][:2]  # Only take x,y coordinates
            if point[0] != 0 or point[1] != 0:  # Check if point is valid
                self.tracking_points['lead_extremities'].append(point)
        
        # Add follow extremities
        for idx in extremity_indices:
            point = follow_pose[idx][:2]
            if point[0] != 0 or point[1] != 0:
                self.tracking_points['follow_extremities'].append(point)

    def update_camera_from_tracks(self, pred_tracks, pred_visibility):
        """Update camera pose based on tracked points"""
        # Split tracked points by category
        num_floor = len(self.tracking_points['floor'])
        num_back = len(self.tracking_points['back_wall'])
        num_left = len(self.tracking_points['left_wall'])
        num_right = len(self.tracking_points['right_wall'])
        
        floor_tracks = pred_tracks[:num_floor]
        back_tracks = pred_tracks[num_floor:num_floor+num_back]
        left_tracks = pred_tracks[num_floor+num_back:num_floor+num_back+num_left]
        right_tracks = pred_tracks[num_floor+num_back+num_left:num_floor+num_back+num_left+num_right]
        
        # Calculate camera adjustments based on point movements
        rotation = Rotation.from_quat(self.camera_poses['rotation'])

        #TODO camera should iteratively pan, zoom, roll, and tilt (in that order)
        #TODO while it is moving, the loss function should recalculate the mean distances, and the rotation and zoom
        #TODO should stop when the loss has been minimized
        if len(back_tracks) > 0:
            back_movement = np.mean(back_tracks[:, -1, 0] - back_tracks[:, 0, 0])
            pan_adjustment = -np.arctan2(back_movement, self.frame_width) * 0.5
            pan_rot = Rotation.from_euler('y', pan_adjustment)
            rotation = pan_rot * rotation
        
        # Update tilt based on vertical movement of back wall points
        if len(back_tracks) > 0:
            back_vertical = np.mean(back_tracks[:, -1, 1] - back_tracks[:, 0, 1])
            tilt_adjustment = np.arctan2(back_vertical, self.frame_height) * 0.5
            tilt_rot = Rotation.from_euler('x', tilt_adjustment)
            rotation = tilt_rot * rotation
        
        # Update roll based on left/right wall point pairs
        if len(left_tracks) > 0 and len(right_tracks) > 0:
            left_vertical = np.mean(left_tracks[:, -1, 1] - left_tracks[:, 0, 1])
            right_vertical = np.mean(right_tracks[:, -1, 1] - right_tracks[:, 0, 1])
            roll_adjustment = np.arctan2(right_vertical - left_vertical, self.frame_width) * 0.5
            roll_rot = Rotation.from_euler('z', roll_adjustment)
            rotation = roll_rot * rotation
        
        # Update zoom based on point separation
        if len(back_tracks) > 1:
            initial_separations = np.linalg.norm(back_tracks[:, 0, :2].reshape(-1, 1, 2) - 
                                               back_tracks[:, 0, :2].reshape(1, -1, 2), axis=2)
            final_separations = np.linalg.norm(back_tracks[:, -1, :2].reshape(-1, 1, 2) - 
                                             back_tracks[:, -1, :2].reshape(1, -1, 2), axis=2)
            
            zoom_ratio = np.mean(final_separations / np.maximum(initial_separations, 1e-6))
            self.camera_poses['focal_length'] *= zoom_ratio
        
        # Update camera pose
        self.camera_poses['rotation'] = rotation.as_quat()

    def check_track_discontinuity(self):
        """Check if either lead or follow track is missing in current frame"""
        current_poses = self.pose_detections.get(str(self.current_frame_idx), {})
        return (self.lead_track_id not in current_poses or 
                self.follow_track_id not in current_poses)

    def prepare_display_frame(self):
        """Prepare frame for display during tracking"""
        display_frame = self.current_frame.copy()
        
        # Draw virtual room
        display_frame = self.virtualRoom.draw_virtual_room(display_frame, self.camera_poses)
        
        # Draw poses
        display_frame = self.draw_poses(display_frame)
        
        # First draw all original points in grey
        points = cv2.goodFeaturesToTrack(
            cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY),
            maxCorners=500,
            qualityLevel=0.05,
            minDistance=20,
            blockSize=3,
            useHarrisDetector=True,
            k=0.04
        )
        
        if points is not None:
            points = points.reshape(-1, 2)
            # Draw original points in grey
            for i, point in enumerate(points):
                pixel_pos = tuple(map(int, point))
                cv2.circle(display_frame, pixel_pos, 3, (128, 128, 128), -1)  # Grey dot
                cv2.putText(display_frame, f"o{i}", 
                          (pixel_pos[0] + 5, pixel_pos[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
        
        # Draw categorized tracking points with larger circles and labels
        point_colors = {
            'floor': (0, 0, 255),        # Red
            'back_wall': (0, 255, 0),    # Green
            'left_wall': (255, 0, 0),    # Blue
            'right_wall': (255, 255, 0), # Cyan
            'lead_extremities': (0, 255, 0),  # Green
            'follow_extremities': (0, 0, 255) # Red
        }
        
        # Draw categorized points
        for category, points in self.tracking_points.items():
            color = point_colors[category]
            for i, point in enumerate(points):
                pixel_pos = tuple(map(int, point))
                # Draw larger circles for categorized points
                cv2.circle(display_frame, pixel_pos, 4, color, 1)  # Hollow circle
                cv2.putText(display_frame, f"{category[0]}{i}", 
                          (pixel_pos[0] + 5, pixel_pos[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # Draw line from original point to projected point if available
                world_point = self.virtualRoom.project_point_to_planes(point, self.camera_poses)
                if world_point is not None:
                    # Project world point back to image
                    rotation = Rotation.from_quat(self.camera_poses['rotation'])
                    point_cam = rotation.inv().apply(world_point - self.camera_poses['position'])
                    if point_cam[2] > 0:  # In front of camera
                        fx = fy = min(self.frame_height, self.frame_width)
                        cx, cy = self.frame_width/2, self.frame_height/2
                        x = -point_cam[0] / point_cam[2]
                        y = -point_cam[1] / point_cam[2]
                        proj_x = int(x * fx * self.camera_poses['focal_length'] + cx)
                        proj_y = int(y * fx * self.camera_poses['focal_length'] + cy)
                        
                        # Draw projected point
                        cv2.circle(display_frame, (proj_x, proj_y), 2, color, -1)
                        cv2.line(display_frame, pixel_pos, (proj_x, proj_y), color, 1)
        
        # Add point counts in top-left corner
        y_offset = 30
        cv2.putText(display_frame, f"Frame: {self.current_frame_idx}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for category, points in self.tracking_points.items():
            color = point_colors[category]
            cv2.putText(display_frame, f"{category}: {len(points)} points", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        return display_frame
