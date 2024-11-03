import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
import os
import json

import utils
from virtual_room import VirtualRoom
import surface_detector


class DanceRoomTracker:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.initial_camera_pose_json_path = output_dir + '/initial_camera_pose.json'
        self.camera_tracking_json_path = output_dir + '/camera_tracking.json'

        self.surfaceDetector = surface_detector.SurfaceDetector(output_dir + "/depth")
        
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

        self.virtualRoom = VirtualRoom()
        
        # Tracking state
        self.current_frame_idx = 0

        # Load VO data at startup
        self.vo_data = self.load_vo_data()
        self.processed_vo_rotations = {}
        self.processed_vo_focal_lengths = {}
        
        # Add tracking for keypoints
        self.rotation_keypoints = {}  # frame_idx -> rotation quaternion
        self.focal_keypoints = {}     # frame_idx -> focal length
        
        # Always store frame 0 as first keypoint
        if self.initial_camera_pose:
            self.rotation_keypoints[0] = self.initial_camera_pose['rotation'].copy()
            self.focal_keypoints[0] = self.initial_camera_pose['focal_length']

        self.frame_height, self.frame_width = None, None
        self.current_frame = None

        # Add timeline UI properties
        self.timeline_height = 50
        self.timeline_margin = 20
        self.scrubber_width = 10
        self.total_frames = self.get_total_frames()
        
        # Track if we're dragging the timeline
        self.dragging_timeline = False
        self.mouse_x = 0
        self.mouse_y = 0

        # Add variables to track pre-movement state
        self.pre_movement_rotation = None
        self.pre_movement_focal = None
        self.camera_has_moved = False

    def draw_timeline(self, frame):
        """Draw timeline scrubber with keyframe markers"""
        h, w = frame.shape[:2]
        timeline_y = h - self.timeline_height
        
        # Draw timeline background
        cv2.rectangle(frame, 
                     (self.timeline_margin, timeline_y),
                     (w - self.timeline_margin, h - self.timeline_margin),
                     (50, 50, 50), -1)
        
        # Draw keyframe markers
        timeline_width = w - 2 * self.timeline_margin
        for keyframe in sorted(self.rotation_keypoints.keys()):
            x = int(self.timeline_margin + (keyframe / self.total_frames) * timeline_width)
            cv2.rectangle(frame,
                         (x - 2, timeline_y),
                         (x + 2, h - self.timeline_margin),
                         (0, 128, 255), -1)
        
        # Draw current frame marker
        current_x = int(self.timeline_margin + 
                       (self.current_frame_idx / self.total_frames) * timeline_width)
        cv2.rectangle(frame,
                     (current_x - self.scrubber_width//2, timeline_y),
                     (current_x + self.scrubber_width//2, h - self.timeline_margin),
                     (0, 255, 0), -1)
    
    def timeline_click_to_frame(self, x, y):
        """Convert timeline click to frame number"""
        timeline_y = self.frame_height - self.timeline_height
        if y < timeline_y:
            return None
            
        timeline_width = self.frame_width - 2 * self.timeline_margin
        relative_x = x - self.timeline_margin
        if 0 <= relative_x <= timeline_width:
            return int((relative_x / timeline_width) * self.total_frames)
        return None

    @staticmethod
    def interpolate_error_ratio(ratio, t):
        """
        Interpolate error ratio towards 1.0 based on distance (t).
        t=1 means use full ratio, t=0 means no change (1.0)
        """
        return 1.0 + (ratio - 1.0) * t

    @staticmethod
    def calculate_rotation_error(old_rot, new_rot):
        """
        Calculate rotation error as a single quaternion transformation.
        Returns the quaternion that transforms old_rot to new_rot.
        """
        # Convert to Rotation objects if they aren't already
        old_r = Rotation.from_quat(old_rot)
        new_r = Rotation.from_quat(new_rot)
        
        # Calculate the difference rotation (error)
        # error_rot = new_rot * old_rot^(-1)
        error_rot = new_r * old_r.inv()
        
        return error_rot.as_quat()

    @staticmethod
    def interpolate_rotation(error_quat, t):
        """
        Interpolate rotation error based on distance (t).
        t=1 means use full rotation, t=0 means no rotation
        """
        # Create array of rotations for slerp
        rotations = Rotation.from_quat(np.array([[0, 0, 0, 1], error_quat]))  # Array of two quaternions
        
        # Use slerp to interpolate between identity (no rotation) and error rotation
        interpolated = Slerp([0, 1], rotations)(t)
        
        return interpolated.as_quat()

    def warp(self):
        print(f"Creating keyframe at frame {self.current_frame_idx}")

        # Calculate rotation error as single quaternion transformation
        rotation_error = self.calculate_rotation_error(
            self.pre_movement_rotation,
            self.frame_rotations[self.current_frame_idx]
        )
        focal_ratio = self.frame_focal_lengths[self.current_frame_idx] / self.pre_movement_focal

        print(f"Rotation error quat: {rotation_error}")
        print(f"Focal ratio: {focal_ratio}")

        # Store as keypoint
        self.rotation_keypoints[self.current_frame_idx] = self.frame_rotations[self.current_frame_idx].copy()
        self.focal_keypoints[self.current_frame_idx] = self.frame_focal_lengths[self.current_frame_idx]

        # Get sorted keyframes
        keyframes = sorted(self.rotation_keypoints.keys())
        new_kf_index = keyframes.index(self.current_frame_idx)

        # Warp frames between previous keyframe and this one
        if new_kf_index > 0:
            prev_keyframe = keyframes[new_kf_index - 1]
            for frame in range(prev_keyframe + 1, self.current_frame_idx):
                # Calculate interpolation factor (0 at prev keyframe, 1 at new keyframe)
                t = (frame - prev_keyframe) / (self.current_frame_idx - prev_keyframe)

                # Get current frame's rotation
                frame_rot = Rotation.from_quat(self.frame_rotations[frame])
                
                # Get interpolated error rotation
                interpolated_error = Rotation.from_quat(self.interpolate_rotation(rotation_error, t))
                
                # Apply interpolated error to frame's rotation
                new_rot = interpolated_error * frame_rot
                
                # Store warped values
                self.frame_rotations[frame] = new_rot.as_quat()

                # Apply interpolated focal length adjustment
                focal_t = self.interpolate_error_ratio(focal_ratio, t)
                self.frame_focal_lengths[frame] *= focal_t

        # If there's a next keyframe, warp frames between this one and next
        if new_kf_index < len(keyframes) - 1:
            next_keyframe = keyframes[new_kf_index + 1]
            for frame in range(self.current_frame_idx + 1, next_keyframe):
                # Calculate reverse interpolation factor (1 at new keyframe, 0 at next keyframe)
                t = 1.0 - ((frame - self.current_frame_idx) / (next_keyframe - self.current_frame_idx))

                # Get current frame's rotation
                frame_rot = Rotation.from_quat(self.frame_rotations[frame])
                
                # Get interpolated error rotation
                interpolated_error = Rotation.from_quat(self.interpolate_rotation(rotation_error, t))
                
                # Apply interpolated error to frame's rotation
                new_rot = interpolated_error * frame_rot
                
                # Store warped values
                self.frame_rotations[frame] = new_rot.as_quat()

                # Apply interpolated focal length adjustment
                focal_t = self.interpolate_error_ratio(focal_ratio, t)
                self.frame_focal_lengths[frame] *= focal_t

    def run_video_loop(self):
        """Updated video loop with timeline interaction"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        # Setup mouse callback
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                frame_idx = self.timeline_click_to_frame(x, y)
                if frame_idx is not None:
                    self.dragging_timeline = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, self.current_frame = cap.read()
                    self.current_frame_idx = frame_idx
                    self.update_display(True)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging_timeline:
                frame_idx = self.timeline_click_to_frame(x, y)
                if frame_idx is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, self.current_frame = cap.read()
                    self.current_frame_idx = frame_idx
                    self.update_display(True)
            
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging_timeline = False
        
        cv2.namedWindow('Dance Room Tracker')
        cv2.setMouseCallback('Dance Room Tracker', mouse_callback)
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        self.frame_height, self.frame_width = frame.shape[:2]
        self.current_frame = frame
        self.virtualRoom.set_frame(self.frame_height, self.frame_width)

        # Create window and set mouse callback
        cv2.namedWindow('Dance Room Tracker')
        self.process_vo_data()
        if len(self.frame_rotations) == 0:
            # first time. set rotations from processed vo data
            self.set_rotations_from_processed_vo()

        # Automatically create a keyframe at the last frame
        last_frame = len(self.vo_data) - 1
        self.rotation_keypoints[last_frame] = self.frame_rotations[last_frame].copy()
        self.focal_keypoints[last_frame] = self.frame_focal_lengths[last_frame]

        self.update_display(True)
        
        playing = False
        
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.save_camera_tracking()
                print("Saved tracking data")

            elif key == ord('d'):
                lines = self.surfaceDetector.find_wall_floor_intersections_for_frame(
                    self.current_frame_idx,
                    self.frame_focal_lengths[self.current_frame_idx],
                    self.frame_width,
                    self.frame_height)
                self.update_display(True, lines)
            
            elif key == ord(' '):  # Toggle play/pause
                playing = not playing
                print(f"Playback: {'Playing' if playing else 'Paused'}")
                self.camera_has_moved = False  # Reset movement flag when playing/pausing
            
            elif key == 13:  # Enter key - create new keyframe
                if self.current_frame_idx > 0 and self.camera_has_moved:  # Only if camera has moved
                    self.warp()
                    
                    print(f"Created keyframe and warped surrounding frames")
                    self.update_display(True)
                    self.save_camera_tracking()
                    self.camera_has_moved = False  # Reset movement flag

            elif key == 83 or playing: # FORWARD
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                self.current_frame = frame
                self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.update_display(not playing)
                self.camera_has_moved = False  # Reset movement flag when changing frames
            
            elif key == 81:  # REVERSE
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_pos = max(0, current_pos - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret:
                    self.current_frame = frame
                    self.current_frame_idx = new_pos
                    self.update_display(not playing)
                    self.camera_has_moved = False  # Reset movement flag when changing frames

            elif not playing:  # Handle inputs when paused
                if self.handle_camera_movement(key):
                    self.camera_has_moved = True
                    self.update_display(True)
            
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()

    #region DRAW

    def project_line_to_2d(self, line):
        """Project a 3D line onto the 2D image plane by finding view frustum intersections"""
        point_on_line, direction, is_floor_wall = line  # Unpack all three values
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Define camera frustum planes (in camera space)
        # Near and far planes
        z_near = 0.1
        z_far = 100.0
        
        # Calculate frustum angles based on focal length
        aspect = self.frame_width / self.frame_height
        fov_y = 2 * np.arctan(0.5 / self.frame_focal_lengths[self.current_frame_idx])
        fov_x = fov_y * aspect
        
        # Frustum planes normals (pointing inward)
        frustum_planes = [
            # Near plane
            ([0, 0, z_near], [0, 0, 1]),
            # Far plane
            ([0, 0, z_far], [0, 0, -1]),
            # Left plane
            ([0, 0, 0], [np.cos(fov_x/2), 0, np.sin(fov_x/2)]),
            # Right plane
            ([0, 0, 0], [-np.cos(fov_x/2), 0, np.sin(fov_x/2)]),
            # Top plane
            ([0, 0, 0], [0, -np.cos(fov_y/2), np.sin(fov_y/2)]),
            # Bottom plane
            ([0, 0, 0], [0, np.cos(fov_y/2), np.sin(fov_y/2)])
        ]
        
        # Find intersections with frustum planes
        intersections = []
        for plane_point, plane_normal in frustum_planes:
            # Line-plane intersection
            denom = np.dot(direction, plane_normal)
            if abs(denom) > 1e-6:  # Check if not parallel
                t = (np.dot(plane_normal, np.array(plane_point) - point_on_line)) / denom
                intersection = point_on_line + direction * t
                
                # Check if intersection is within frustum bounds
                if z_near <= intersection[2] <= z_far:
                    # Check horizontal bounds
                    x_bound = intersection[2] * np.tan(fov_x/2)
                    if -x_bound <= intersection[0] <= x_bound:
                        # Check vertical bounds
                        y_bound = intersection[2] * np.tan(fov_y/2)
                        if -y_bound <= intersection[1] <= y_bound:
                            intersections.append(intersection)
        
        if len(intersections) >= 2:
            # Sort intersections by distance along the line
            distances = [np.dot(p - point_on_line, direction) for p in intersections]
            sorted_indices = np.argsort(distances)
            start_point = intersections[sorted_indices[0]]
            end_point = intersections[sorted_indices[-1]]
            
            # Project the valid intersection points
            points_2d, behind = self.virtualRoom.project_points(
                np.array([start_point, end_point]),
                Rotation.from_quat([0,0,0,1]),
                [0,0,0],
                self.frame_focal_lengths[self.current_frame_idx]
            )
            
            if len(points_2d) >= 2:
                # Clip line to frame bounds
                x0, y0 = points_2d[0]
                x1, y1 = points_2d[1]
                
                # Cohen-Sutherland line clipping
                def clip_line(x0, y0, x1, y1, w, h):
                    def compute_code(x, y, w, h):
                        code = 0
                        if x < 0: code |= 1
                        if x >= w: code |= 2
                        if y < 0: code |= 4
                        if y >= h: code |= 8
                        return code
                    
                    # Compute codes for both endpoints
                    code0 = compute_code(x0, y0, w, h)
                    code1 = compute_code(x1, y1, w, h)
                    
                    while True:
                        if not (code0 | code1):  # Both endpoints inside
                            return (int(x0), int(y0)), (int(x1), int(y1))
                        elif code0 & code1:  # Both endpoints outside same region
                            return None
                        
                        # Pick an endpoint outside the clip window
                        code = code0 if code0 else code1
                        
                        # Find intersection point
                        if code & 1:  # Left
                            y = y0 + (y1 - y0) * (-x0) / (x1 - x0)
                            x = 0
                        elif code & 2:  # Right
                            y = y0 + (y1 - y0) * (w - x0) / (x1 - x0)
                            x = w - 1
                        elif code & 4:  # Top
                            x = x0 + (x1 - x0) * (-y0) / (y1 - y0)
                            y = 0
                        elif code & 8:  # Bottom
                            x = x0 + (x1 - x0) * (h - y0) / (y1 - y0)
                            y = h - 1
                        
                        # Replace outside point
                        if code == code0:
                            x0, y0 = x, y
                            code0 = compute_code(x0, y0, w, h)
                        else:
                            x1, y1 = x, y
                            code1 = compute_code(x1, y1, w, h)
                
                # Clip line to frame bounds
                clipped = clip_line(x0, y0, x1, y1, self.frame_width, self.frame_height)
                if clipped:
                    return [clipped[0], clipped[1]]
        
        return []

    def draw_intersection_lines(self, display_frame, lines):
        """Draw the intersection lines with proper clipping and color coding"""
        for line in lines:
            points_2d = self.project_line_to_2d(line)
            if len(points_2d) == 2:  # Only draw if we have valid start and end points
                point_on_line, direction, is_floor_wall = line
                
                # Color: red for floor-wall intersections, green for wall-wall
                color = (0, 0, 255) if is_floor_wall else (0, 255, 0)
                
                # Draw the line segment
                cv2.line(display_frame, points_2d[0], points_2d[1], color, 2)

    def draw_point_cloud(self, display_frame, floor_data, wall_data):
        """Draw color-coded point cloud with depth-based intensity"""
        floor_points, floor_depths = floor_data
        wall_points, wall_depths = wall_data

        def get_depth_intensity(depth, min_depth=0.5, max_depth=30.0):
            """Convert depth to color intensity (0-255)"""
            depth = np.clip(depth, min_depth, max_depth)
            return int(255 * (1 - (depth - min_depth) / (max_depth - min_depth)))

        # Draw points with larger radius for better visibility
        point_radius = max(1, int(self.frame_width / 640))  # Scale point size with frame width

        # Project floor points
        if len(floor_points) > 0:
            proj_floor_pts, floor_behind = self.virtualRoom.project_points(
                floor_points,
                Rotation.from_quat([0,0,0,1]),
                [0,0,0],
                self.frame_focal_lengths[self.current_frame_idx])
            
            # Draw floor points with depth-based intensity
            for i, point in enumerate(proj_floor_pts):
                intensity = get_depth_intensity(floor_depths[i])
                cv2.circle(display_frame, point, point_radius, (0, 0, intensity), -1)

        # Project wall points
        if len(wall_points) > 0:
            proj_wall_pts, wall_behind = self.virtualRoom.project_points(
                wall_points,
                Rotation.from_quat([0,0,0,1]),
                [0,0,0],
                self.frame_focal_lengths[self.current_frame_idx])
            
            # Draw wall points with depth-based intensity
            for i, point in enumerate(proj_wall_pts):
                intensity = get_depth_intensity(wall_depths[i])
                cv2.circle(display_frame, point, point_radius, (0, intensity, 0), -1)

    def update_display(self, paused, lines=None):
        """Update display with current frame and overlays"""
        display_frame = self.current_frame.copy()

        # Draw virtual room with current frame's rotation and focal length
        display_frame = self.virtualRoom.draw_virtual_room(
            display_frame,
            self.initial_camera_pose['position'],
            Rotation.from_quat(self.frame_rotations[self.current_frame_idx]),
            self.frame_focal_lengths[self.current_frame_idx]
        )

        if lines is not None:
            intersection_lines, floor_data, wall_data = lines
            # Draw point cloud first (so lines appear on top)
            self.draw_point_cloud(display_frame, floor_data, wall_data)
            # Draw intersection lines
            self.draw_intersection_lines(display_frame, intersection_lines)

        # Add UI text
        if paused:
            if self.current_frame_idx == 0:
                cv2.putText(display_frame,
                            "WASD: Move camera | QE: Up/Down | Arrows: Pan/Tilt | OP: Roll | ZX: Zoom",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame,
                            "Arrows: Pan/Tilt | OP: Roll | ZX: Zoom | Enter: Set Keyframe",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw frame counter and keyframe indicator
        text = f"Frame: {self.current_frame_idx}"
        if self.current_frame_idx in self.rotation_keypoints:
            text += " (Keyframe)"
        cv2.putText(display_frame, text,
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw timeline
        self.draw_timeline(display_frame)

        cv2.imshow('Dance Room Tracker', display_frame)

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
        pos_delta = 0.1
        current_position = None
        current_rotation = None
        current_focal_length = None
        update = False

        if key == ord('w') or key == ord('s') or key == ord('a') or key == ord('d') or key == ord('q') or key == ord('e') or key == ord('j') or key == ord('k') or key == ord('i') or key == ord('l') or key == ord('z') or key == ord('x') or key == ord('o') or key == ord('p'):
            update = True
            current_position = self.initial_camera_pose['position']
            current_rotation = self.initial_camera_pose['rotation']
            current_focal_length = self.initial_camera_pose['focal_length']

        if key == ord('w'):
            self.initial_camera_pose['position'][2] = current_position[2] + pos_delta
        elif key == ord('z'):  # Increase focal length
            self.frame_focal_lengths[self.current_frame_idx] = current_focal_length + focal_delta
            self.initial_camera_pose['focal_length'] = self.frame_focal_lengths[self.current_frame_idx]
        elif key == ord('x'):  # Decrease focal length
            self.frame_focal_lengths[self.current_frame_idx] = max(0.1, current_focal_length - focal_delta)
            self.initial_camera_pose['focal_length'] = self.frame_focal_lengths[self.current_frame_idx]
        elif key == ord('i'):  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
        elif key == ord('k'):  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
        elif key == ord('j'):  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
        elif key == ord('l'):  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
            self.initial_camera_pose['rotation'] = self.frame_rotations[self.current_frame_idx]

        if update:
            self.save_initial_camera_pose()

        return update

    def _handle_orientation_only(self, key):
        """Handle only rotation and focal length changes"""
        rot_delta = 0.005
        focal_delta = 0.01

        current_rotation = None
        current_focal_length = None
        update = False

        if key == ord('j') or key == ord('k') or key == ord('i') or key == ord('l') or key == ord('z') or key == ord('x') or key == ord('o') or key == ord('p'):
            current_rotation = Rotation.from_quat(self.frame_rotations[self.current_frame_idx])
            current_focal_length = self.frame_focal_lengths[self.current_frame_idx]
            update = True
            if not self.camera_has_moved:
                self.pre_movement_rotation = self.frame_rotations[self.current_frame_idx].copy()
                self.pre_movement_focal = self.frame_focal_lengths[self.current_frame_idx]

        if key == ord('z'):  # Increase focal length
            self.frame_focal_lengths[self.current_frame_idx] = current_focal_length + focal_delta
        elif key == ord('x'):  # Decrease focal length
            self.frame_focal_lengths[self.current_frame_idx] = max(0.1, current_focal_length - focal_delta)
        elif key == ord('i'):  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('k'):  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('j'):  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('l'):  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()

        return update

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

        # Get initial camera state
        initial_rotation = Rotation.from_quat(self.initial_camera_pose['rotation'])
        initial_focal = self.initial_camera_pose['focal_length']
        initial_z = self.vo_data[0][2]  # Initial z position from VO

        # Process each frame
        window_size = 20
        raw_rotations = []  # Store raw rotation matrices for smoothing

        # First pass: calculate raw rotations
        for frame_data in self.vo_data:
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
        for frame_idx, frame_data in enumerate(self.vo_data):
            # Store smoothed rotation
            self.processed_vo_rotations[frame_idx] = smoothed_rotations[frame_idx].as_quat()

            # Process focal length as before
            pos = np.array(frame_data[:3])
            z_delta = pos[2] - initial_z
            focal_delta = z_delta * 0.7  # RATIO of Z to Focal Length
            self.processed_vo_focal_lengths[frame_idx] = initial_focal + focal_delta

    def set_rotations_from_processed_vo(self):
        for frame_idx, frame_data in enumerate(self.vo_data):
            self.frame_rotations[frame_idx] = self.processed_vo_rotations[frame_idx]
            self.frame_focal_lengths[frame_idx] = self.processed_vo_focal_lengths[frame_idx]

        print("set rotations and focal lengths from adjusted visual odometry")

    def get_total_frames(self):
        """Get total number of frames in video"""
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    # endregion