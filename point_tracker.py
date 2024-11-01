import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from cotracker_runner import CoTracker

#TODO saved this code for cotracking frames
class PointTracker:
    def __init__(self, video_path, output_dir, pose_detections, virtualRoom):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cotracker = CoTracker()
        self.pose_detections = pose_detections
        self.virtualRoom = virtualRoom

        self.frame_height, self.frame_width = None, None
        self.current_frame = None

        # Replace tracking_points with simple category mapping - remove floor
        self.point_categories = {
            'back_wall': set(),
            'left_wall': set(),
            'right_wall': set()
        }

        self.next_track_id = 0  # For generating unique IDs

        self.point_visibility = {}  # ID -> bool
        self.point_tracks = {}  # ID -> {track: array, visibility: array, start_frame: int}

        self.track_display_frames = 50  # Number of frames to display in past/future

        # Store initial world positions of tracking points
        self.initial_world_positions = {}  # point_id -> (surface, world_pos)

        # Joint keypoint indices for tracking
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # All COCO joints

    def run(self, camera_poses):
        print("Running initial point tracking...")
        success = self.track_points_chunk(camera_poses)  # Check if tracking succeeded
        if not success:
            print("Initial tracking failed!")
            return

    def video_loop(self):
        while True:
            # Check remaining track length and run new tracking if needed
            min_future_frames = float('inf')
            for track_data in self.point_tracks.values():
                track_end = track_data['start_frame'] + len(track_data['track'])
                remaining_frames = track_end - self.current_frame_idx
                min_future_frames = min(min_future_frames, remaining_frames)

            print(f"\rFrame {self.current_frame_idx}, Future frames: {min_future_frames}", end='', flush=True)

            if min_future_frames < 10:
                print(f"\nRunning new tracking at frame {self.current_frame_idx}")
                if not self.track_points_chunk(None): #TODO send camera_poses
                    print("Tracking failed, stopping")
                    break

        self.update_camera_from_tracks()

    def track_points_chunk(self, camera_poses):
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
            self.find_new_points(camera_poses)

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

    def find_new_points(self, pose_detections, virtualRoom, camera_poses):
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
        poses = pose_detections[str(self.current_frame_idx)]
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
                                intersection = virtualRoom.project_point_to_planes(point, camera_poses)
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
                camera_to_wall = abs(current_params['position'][2] + self.virtualRoom.room_depth / 2)
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
                        camera_to_wall = abs(current_params['position'][0] + self.virtualRoom.room_width / 2)
                        z_angles = np.arctan2(z_diffs, camera_to_wall)
                    else:  # right wall
                        camera_to_wall = abs(current_params['position'][0] - self.virtualRoom.room_width / 2)
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
                        camera_to_wall = abs(current_params['position'][2] + self.virtualRoom.room_depth / 2)
                    elif wall == 'left_wall':
                        camera_to_wall = abs(current_params['position'][0] + self.virtualRoom.room_width / 2)
                    else:  # right wall
                        camera_to_wall = abs(current_params['position'][0] - self.virtualRoom.room_width / 2)

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
                    cx, cy = self.frame_width / 2, self.frame_height / 2
                    x = -point_cam[0] / point_cam[2]
                    y = -point_cam[1] / point_cam[2]
                    proj_x = x * fx * test_focal + cx
                    proj_y = y * fy * test_focal + cy

                    # Calculate error between projected initial position and current tracked position
                    dx = proj_x - current_point[0]
                    dy = proj_y - current_point[1]
                    total_error += dx * dx + dy * dy
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

            print(
                f"Roll-Zoom optimization: best_error={best_error:.2f}, roll={best_params[0]:.4f}, zoom={best_params[1]:.4f}")
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

    def prepare_display_frame(self):
        """Prepare frame for display during tracking"""
        current_params = self.get_current_camera_params()

        display_frame = self.current_frame.copy()
        display_frame = self.virtualRoom.draw_virtual_room(display_frame, current_params)

        point_colors = {
            'back_wall': (0, 255, 0),  # Green
            'left_wall': (255, 0, 0),  # Blue
            'right_wall': (255, 255, 0),  # Cyan
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
                            cx, cy = self.frame_width / 2, self.frame_height / 2
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