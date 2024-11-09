import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
import os
import json

import utils
from virtual_room import VirtualRoom


class DanceRoomTracker:
    def __init__(self, video_path:str, output_dir:str, room_dimension):
        self.__video_path = video_path
        self.__output_dir = output_dir
        self.__initial_camera_pose_json_path = os.path.join(output_dir, 'initial_camera_pose.json')
        self.__camera_tracking_json_path = os.path.join(output_dir, 'camera_tracking.json')
        
        # Load initial camera pose (position and initial orientation/focal)
        self.__user_has_set_cam_pos = False
        self.__initial_camera_pose = self.__load_initial_camera_pose()
        
        # Per-frame camera tracking data
        self.__frame_rotations = {}  # frame_idx -> quaternion
        self.__frame_focal_lengths = {}  # frame_idx -> focal_length
        
        # Load existing tracking data if it exists
        tracking_data = utils.load_json(self.__camera_tracking_json_path) or {}
        for frame_str, data in tracking_data.items():
            frame_idx = int(frame_str)
            self.__frame_rotations[frame_idx] = np.array(data['rotation'])
            self.__frame_focal_lengths[frame_idx] = data['focal_length']

        self.__virtualRoom = VirtualRoom(room_dimension)
        
        # Tracking state
        self.current_frame_idx = 0

        # Load VO data at startup
        self.__vo_data = self.__load_vo_data()
        self.__processed_vo_rotations = {}
        self.__processed_vo_focal_lengths = {}
        
        # Add tracking for keypoints
        self.__rotation_keypoints = {}  # frame_idx -> rotation quaternion
        self.__focal_keypoints = {}     # frame_idx -> focal length

        # store cam initial keypoint
        if self.__initial_camera_pose:
            self.__rotation_keypoints[self.__initial_camera_pose['frame_num']] = self.__initial_camera_pose['rotation'].copy()
            self.__focal_keypoints[self.__initial_camera_pose['frame_num']] = self.__initial_camera_pose['focal_length']

        self.__frame_height, self.__frame_width = None, None
        self.current_frame = None

        # Add timeline UI properties
        self.__timeline_height = 50
        self.__timeline_margin = 20
        self.__scrubber_width = 10
        self.__total_frames = self.__get_total_frames()

        self.__dragging_timeline = False
        self.__camera_has_moved = False

    def __draw_timeline(self, frame):
        """Draw timeline scrubber with keyframe markers"""
        h, w = frame.shape[:2]
        timeline_y = h - self.__timeline_height
        
        # Draw timeline background
        cv2.rectangle(frame, 
                     (self.__timeline_margin, timeline_y),
                     (w - self.__timeline_margin, h - self.__timeline_margin),
                     (50, 50, 50), -1)
        
        # Draw keyframe markers
        timeline_width = w - 2 * self.__timeline_margin
        for keyframe in sorted(self.__rotation_keypoints.keys()):
            x = int(self.__timeline_margin + (keyframe / self.__total_frames) * timeline_width)
            cv2.rectangle(frame,
                          (x - 2, timeline_y),
                          (x + 2, h - self.__timeline_margin),
                          (0, 128, 255), -1)

        # Draw current frame marker
        current_x = int(self.__timeline_margin +
                        (self.current_frame_idx / self.__total_frames) * timeline_width)
        cv2.rectangle(frame,
                      (current_x - self.__scrubber_width // 2, timeline_y),
                      (current_x + self.__scrubber_width // 2, h - self.__timeline_margin),
                      (0, 255, 0), -1)

    def __timeline_click_to_frame(self, x, y):
        """Convert timeline click to frame number"""
        timeline_y = self.__frame_height - self.__timeline_height
        if y < timeline_y:
            return None
            
        timeline_width = self.__frame_width - 2 * self.__timeline_margin
        relative_x = x - self.__timeline_margin
        if 0 <= relative_x <= timeline_width:
            return int((relative_x / timeline_width) * self.__total_frames)
        return None

    @staticmethod
    def __interpolate_error_ratio(ratio, t):
        """
        Interpolate error ratio towards 1.0 based on distance (t).
        t=1 means use full ratio, t=0 means no change (1.0)
        """
        return 1.0 + (ratio - 1.0) * t

    @staticmethod
    def __calculate_rotation_error(old_rot, new_rot):
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
    def __interpolate_rotation(error_quat, t):
        """
        Interpolate rotation error based on distance (t).
        t=1 means use full rotation, t=0 means no rotation
        """
        # Create array of rotations for slerp
        rotations = Rotation.from_quat(np.array([[0, 0, 0, 1], error_quat]))  # Array of two quaternions
        
        # Use slerp to interpolate between identity (no rotation) and error rotation
        interpolated = Slerp([0, 1], rotations)(t)
        
        return interpolated.as_quat()

    def __warp(self):
        print(f"Creating keyframe at frame {self.current_frame_idx}")

        # Calculate rotation error as single quaternion transformation
        rotation_error = self.__calculate_rotation_error(
            self.__pre_movement_rotation,
            self.__frame_rotations[self.current_frame_idx]
        )
        focal_ratio = self.__frame_focal_lengths[self.current_frame_idx] / self.__pre_movement_focal

        print(f"Rotation error quat: {rotation_error}")
        print(f"Focal ratio: {focal_ratio}")

        # Store as keypoint
        self.__rotation_keypoints[self.current_frame_idx] = self.__frame_rotations[self.current_frame_idx].copy()
        self.__focal_keypoints[self.current_frame_idx] = self.__frame_focal_lengths[self.current_frame_idx]

        # Get sorted keyframes
        keyframes = sorted(self.__rotation_keypoints.keys())
        new_kf_index = keyframes.index(self.current_frame_idx)

        # Warp frames between previous keyframe and this one
        if new_kf_index > 0:
            prev_keyframe = keyframes[new_kf_index - 1]
            for frame in range(prev_keyframe + 1, self.current_frame_idx):
                # Calculate interpolation factor (0 at prev keyframe, 1 at new keyframe)
                t = (frame - prev_keyframe) / (self.current_frame_idx - prev_keyframe)

                # Get current frame's rotation
                frame_rot = Rotation.from_quat(self.__frame_rotations[frame])
                
                # Get interpolated error rotation
                interpolated_error = Rotation.from_quat(self.__interpolate_rotation(rotation_error, t))
                
                # Apply interpolated error to frame's rotation
                new_rot = interpolated_error * frame_rot
                
                # Store warped values
                self.__frame_rotations[frame] = new_rot.as_quat()

                # Apply interpolated focal length adjustment
                focal_t = self.__interpolate_error_ratio(focal_ratio, t)
                self.__frame_focal_lengths[frame] *= focal_t

        # If there's a next keyframe, warp frames between this one and next
        if new_kf_index < len(keyframes) - 1:
            next_keyframe = keyframes[new_kf_index + 1]
            for frame in range(self.current_frame_idx + 1, next_keyframe):
                # Calculate reverse interpolation factor (1 at new keyframe, 0 at next keyframe)
                t = 1.0 - ((frame - self.current_frame_idx) / (next_keyframe - self.current_frame_idx))

                # Get current frame's rotation
                frame_rot = Rotation.from_quat(self.__frame_rotations[frame])
                
                # Get interpolated error rotation
                interpolated_error = Rotation.from_quat(self.__interpolate_rotation(rotation_error, t))
                
                # Apply interpolated error to frame's rotation
                new_rot = interpolated_error * frame_rot
                
                # Store warped values
                self.__frame_rotations[frame] = new_rot.as_quat()

                # Apply interpolated focal length adjustment
                focal_t = self.__interpolate_error_ratio(focal_ratio, t)
                self.__frame_focal_lengths[frame] *= focal_t

    def run_video_loop(self):
        """Updated video loop with timeline interaction"""
        cap = cv2.VideoCapture(self.__video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        # Setup mouse callback
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                frame_idx = self.__timeline_click_to_frame(x, y)
                if frame_idx is not None:
                    self.__dragging_timeline = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, self.current_frame = cap.read()
                    self.current_frame_idx = frame_idx
                    self.__update_display(True)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.__dragging_timeline:
                frame_idx = self.__timeline_click_to_frame(x, y)
                if frame_idx is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, self.current_frame = cap.read()
                    self.current_frame_idx = frame_idx
                    self.__update_display(True)
            
            elif event == cv2.EVENT_LBUTTONUP:
                self.__dragging_timeline = False
        
        cv2.namedWindow('Dance Room Tracker')
        cv2.setMouseCallback('Dance Room Tracker', mouse_callback)
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        self.__frame_height, self.__frame_width = frame.shape[:2]
        self.current_frame = frame
        self.__virtualRoom.set_frame(self.__frame_height, self.__frame_width)

        # Create window and set mouse callback
        cv2.namedWindow('Dance Room Tracker')

        def initialize_vo():
            self.__process_vo_data()
            self.__set_rotations_from_processed_vo()

            # Automatically create a keyframe at the last frame
            last_frame = len(self.__vo_data) - 1
            self.__rotation_keypoints[last_frame] = self.__frame_rotations[last_frame].copy()
            self.__focal_keypoints[last_frame] = self.__frame_focal_lengths[last_frame]

        if self.__user_has_set_cam_pos and len(self.__frame_rotations) == 0:
            initialize_vo()

        self.__update_display(True)
        
        playing = False
        
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Toggle play/pause
                playing = not playing
                print(f"Playback: {'Playing' if playing else 'Paused'}")
                self.__camera_has_moved = False  # Reset movement flag when playing/pausing
            
            elif key == 13:  # Enter key - create new keyframe
                if not self.__user_has_set_cam_pos:
                    self.__save_initial_camera_pose()
                    initialize_vo()
                elif self.current_frame_idx > 0 and self.__camera_has_moved:  # Only if camera has moved
                    self.__warp()
                    
                    print(f"Created keyframe and warped surrounding frames")
                    self.__update_display(True)
                    self.__save_camera_tracking()
                    self.__camera_has_moved = False  # Reset movement flag

            elif key == 83 or playing:  # FORWARD
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                self.current_frame = frame
                self.current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.__update_display(not playing)
                self.__camera_has_moved = False  # Reset movement flag when changing frames
            
            elif key == 81:  # REVERSE
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                new_pos = max(0, current_pos - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret:
                    self.current_frame = frame
                    self.current_frame_idx = new_pos
                    self.__update_display(not playing)
                    self.__camera_has_moved = False  # Reset movement flag when changing frames

            elif not playing:  # Handle inputs when paused
                if self.__handle_camera_movement(key):
                    self.__camera_has_moved = True
                    self.__update_display(True)
            
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def __update_display(self, paused):
        """Update display with current frame and overlays"""
        display_frame = self.current_frame.copy()

        rot = Rotation.from_quat(self.__initial_camera_pose['rotation'])
        if not len(self.__frame_rotations) == 0:
            rot = Rotation.from_quat(self.__frame_rotations[self.current_frame_idx])

        focal = self.__initial_camera_pose['focal_length']
        if not len(self.__frame_focal_lengths) == 0:
            focal = self.__frame_focal_lengths[self.current_frame_idx]

        # Draw virtual room with current frame's rotation and focal length
        display_frame = self.__virtualRoom.draw_virtual_room(
            display_frame,
            self.__initial_camera_pose['position'],
            rot,
            focal
        )

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
        if self.current_frame_idx in self.__rotation_keypoints:
            text += " (Keyframe)"
        cv2.putText(display_frame, text,
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw timeline
        self.__draw_timeline(display_frame)

        cv2.imshow('Dance Room Tracker', display_frame)

    # region CAMERA

    def __handle_camera_movement(self, key):
        """Handle keyboard input for camera movement"""
        if not self.__user_has_set_cam_pos:
            # Allow full camera control at frame 0
            return self.__handle_full_camera_movement(key)
        else:
            # Only allow rotation and focal length changes at other frames
            return self.__handle_orientation_only(key)

    def __handle_full_camera_movement(self, key):
        """Handle only rotation and focal length changes"""

        rot_delta = 0.005
        focal_delta = 0.01
        pos_delta = 0.1
        current_position = None
        current_rotation = None
        current_focal_length = None
        update = False

        if key == ord('w') or key == ord('s') or key == ord('a') or key == ord('d') or key == ord('q') or key == ord(
                'e') or key == ord('j') or key == ord('k') or key == ord('i') or key == ord('l') or key == ord(
                'z') or key == ord('x') or key == ord('o') or key == ord('p'):
            update = True
            current_position = self.__initial_camera_pose['position']
            current_rotation = Rotation.from_quat(self.__initial_camera_pose['rotation'])
            current_focal_length = self.__initial_camera_pose['focal_length']

        if key == ord('w'):
            self.__initial_camera_pose['position'][2] = current_position[2] + pos_delta
        elif key == ord('s'):
            self.__initial_camera_pose['position'][2] = current_position[2] - pos_delta
        elif key == ord('a'):
            self.__initial_camera_pose['position'][0] = current_position[0] - pos_delta
        elif key == ord('d'):
            self.__initial_camera_pose['position'][0] = current_position[0] + pos_delta
        elif key == ord('q'):
            self.__initial_camera_pose['position'][1] = current_position[1] - pos_delta
        elif key == ord('e'):
            self.__initial_camera_pose['position'][1] = current_position[1] + pos_delta
        elif key == ord('z'):  # Increase focal length
            self.__initial_camera_pose['focal_length'] = current_focal_length + focal_delta
        elif key == ord('x'):  # Decrease focal length
            self.__initial_camera_pose['focal_length'] = max(0.1, current_focal_length - focal_delta)
        elif key == ord('i'):  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.__initial_camera_pose['rotation'] = (rot * current_rotation).as_quat()
        elif key == ord('k'):  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.__initial_camera_pose['rotation'] = (rot * current_rotation).as_quat()
        elif key == ord('j'):  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.__initial_camera_pose['rotation'] = (rot * current_rotation).as_quat()
        elif key == ord('l'):  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.__initial_camera_pose['rotation'] = (rot * current_rotation).as_quat()
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.__initial_camera_pose['rotation'] = (rot * current_rotation).as_quat()
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.__initial_camera_pose['rotation'] = (rot * current_rotation).as_quat()

        return update

    def __handle_orientation_only(self, key):
        """Handle only rotation and focal length changes"""
        rot_delta = 0.005
        focal_delta = 0.01

        current_rotation = None
        current_focal_length = None
        update = False

        if key == ord('j') or key == ord('k') or key == ord('i') or key == ord('l') or key == ord('z') or key == ord(
                'x') or key == ord('o') or key == ord('p'):
            current_rotation = Rotation.from_quat(self.__frame_rotations[self.current_frame_idx])
            current_focal_length = self.__frame_focal_lengths[self.current_frame_idx]
            update = True
            if not self.__camera_has_moved:
                self.__pre_movement_rotation = self.__frame_rotations[self.current_frame_idx].copy()
                self.__pre_movement_focal = self.__frame_focal_lengths[self.current_frame_idx]

        if key == ord('z'):  # Increase focal length
            self.__frame_focal_lengths[self.current_frame_idx] = current_focal_length + focal_delta
        elif key == ord('x'):  # Decrease focal length
            self.__frame_focal_lengths[self.current_frame_idx] = max(0.1, current_focal_length - focal_delta)
        elif key == ord('i'):  # Up arrow (tilt up)
            rot = Rotation.from_euler('x', -rot_delta)
            self.__frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('k'):  # Down arrow (tilt down)
            rot = Rotation.from_euler('x', rot_delta)
            self.__frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('j'):  # Left arrow (pan left)
            rot = Rotation.from_euler('y', -rot_delta)
            self.__frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('l'):  # Right arrow (pan right)
            rot = Rotation.from_euler('y', rot_delta)
            self.__frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('o'):  # Roll counter-clockwise
            rot = Rotation.from_euler('z', -rot_delta)
            self.__frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()
        elif key == ord('p'):  # Roll clockwise
            rot = Rotation.from_euler('z', rot_delta)
            self.__frame_rotations[self.current_frame_idx] = (rot * current_rotation).as_quat()

        return update

    # endregion

    # region FILE IO

    def __load_initial_camera_pose(self):
        """Load initial camera pose from JSON if it exists"""
        data = utils.load_json(self.__initial_camera_pose_json_path)
        if data:
            self.__user_has_set_cam_pos = True
            return {
                'frame_num': int(data['frame_num']),
                'position': np.array(data['position']),
                'rotation': np.array(data['rotation']),
                'focal_length': data['focal_length']
            }
        else:
            # Default camera pose: at +Z looking towards -Z (back wall)
            # Camera height is now positive since y=0 is at floor level
            return {
                'frame_num': 0,
                'position': np.array([0.0, 1.1, 3.4]),  # Position with positive Y for height above floor
                'rotation': np.array([0.0, 1.0, 0.0, 0.0]),  # Looking towards -Z
                'focal_length': 1.400
            }

    def __save_initial_camera_pose(self):
        """Save camera poses to JSON"""
        data = {
            'frame_num': self.current_frame_idx,
            'position': self.__initial_camera_pose['position'].tolist(),
            'rotation': self.__initial_camera_pose['rotation'].tolist(),
            'focal_length': self.__initial_camera_pose['focal_length']
        }
        utils.save_json(data, self.__initial_camera_pose_json_path)
        self.__user_has_set_cam_pos = True
        print(f'saved to {self.__initial_camera_pose_json_path}')

    def __save_camera_tracking(self):
        """Save per-frame camera tracking data"""
        tracking_data = {}
        for frame_idx in self.__frame_rotations.keys():
            tracking_data[str(frame_idx)] = {
                'rotation': self.__frame_rotations[frame_idx].tolist(),
                'focal_length': self.__frame_focal_lengths[frame_idx]
            }
        utils.save_json(tracking_data, self.__camera_tracking_json_path)
        print(f'Saved camera tracking to {self.__camera_tracking_json_path}')

    def __load_vo_data(self):
        """Load visual odometry data from json"""
        # Construct VO json path
        video_name = os.path.splitext(os.path.basename(self.__video_path))[0]
        vo_path = os.path.join(self.__output_dir, f"{video_name}_vo.json")

        try:
            with open(vo_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: No VO data found at {vo_path}")
            return None

    def __process_vo_data(self):
        """Process visual odometry data after camera calibration"""

        # Get initial camera state
        initial_rotation = Rotation.from_quat(self.__initial_camera_pose['rotation'])
        initial_focal = self.__initial_camera_pose['focal_length']
        initial_z = self.__vo_data[0][2]  # Initial z position from VO

        # Process each frame
        window_size = 20
        raw_rotations = []  # Store raw rotation matrices for smoothing

        # First pass: calculate raw rotations
        for frame_data in self.__vo_data:
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
        for frame_idx, frame_data in enumerate(self.__vo_data):
            # Store smoothed rotation
            self.__processed_vo_rotations[frame_idx] = smoothed_rotations[frame_idx].as_quat()

            # Process focal length as before
            pos = np.array(frame_data[:3])
            z_delta = pos[2] - initial_z
            focal_delta = z_delta * 0.7  # RATIO of Z to Focal Length
            self.__processed_vo_focal_lengths[frame_idx] = initial_focal + focal_delta

    def __set_rotations_from_processed_vo(self):
        for frame_idx, frame_data in enumerate(self.__vo_data):
            self.__frame_rotations[frame_idx] = self.__processed_vo_rotations[frame_idx]
            self.__frame_focal_lengths[frame_idx] = self.__processed_vo_focal_lengths[frame_idx]

        print("set rotations and focal lengths from adjusted visual odometry")

    def __get_total_frames(self):
        """Get total number of frames in video"""
        cap = cv2.VideoCapture(self.__video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    # endregion