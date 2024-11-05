import numpy as np
import os
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import cv2

class SurfaceDetector:
    def __init__(self, depth_dir, pixel_to_meter=0.000264583):
        self.depth_dir = depth_dir
        self.depth_width = 640   # Original depth map dimensions
        self.depth_height = 480
        self.pixel_to_meter = pixel_to_meter

    def load_depth_map(self, frame_num):
        depth_file = os.path.join(self.depth_dir, f'{frame_num:06d}.npz')
        if os.path.exists(depth_file):
            with np.load(depth_file) as data:
                keys = list(data.keys())
                if keys:
                    return data[keys[0]]
                else:
                    print(f"Warning: No data found in {depth_file}")
                    return None
        else:
            print(f"Warning: Depth file not found: {depth_file}")
            return None

    @staticmethod
    def project_point_to_world(image_point, rotation, focal_length, frame_width, frame_height):
        """Project image point to world coordinates and determine which plane it lies on"""
        fx = fy = min(frame_height, frame_width)
        cx, cy = frame_width / 2, frame_height / 2
        rotation = Rotation.from_quat(rotation)

        # Convert to normalized device coordinates - flip X sign to change orientation
        x_ndc = -(image_point[0] - cx) / fx  # Negative sign to flip X orientation
        y_ndc = (cy - image_point[1]) / fy  # Keep Y flipped

        # Create ray in camera space (positive Z is forward)
        ray_dir = np.array([
            x_ndc / focal_length,
            y_ndc / focal_length,
            1.0  # Positive Z for forward
        ])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Transform ray to world space
        world_ray = rotation.apply(ray_dir)

        return world_ray

    def find_wall_floor_intersections_for_frame(self, frame_num, frame_width, frame_height):
        """Find wall-floor and wall-wall intersections using vertical stripe analysis"""
        depth_map = self.load_depth_map(frame_num)
        if depth_map is None:
            return []

        def colorize(depth_map, vmin=None, vmax=None, cmap='magma_r'):
            """Colorize depth map using specified colormap"""
            if vmin is None:
                vmin = depth_map.min()
            if vmax is None:
                vmax = depth_map.max()
                
            normalized = np.clip((depth_map - vmin) / (vmax - vmin), 0, 1)
            colormap = plt.get_cmap(cmap)
            colored = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
            return colored

        # Create debug visualization
        vmax = depth_map.max()
        depth_vis = colorize(depth_map, vmin=0.01, vmax=vmax)
        debug_frame = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)

        # Scale factors to convert from depth map to frame coordinates
        scale_x = frame_width / depth_map.shape[1]
        scale_y = frame_height / depth_map.shape[0]

        # Parameters for analysis
        min_points = 10
        initial_points = 10
        outlier_threshold = 0.15
        stripe_width = 5

        # Parameters for wall slope analysis
        edge_width = 50  # Width of edge region to analyze
        slope_difference_threshold = 0.3  # Minimum difference in slopes to indicate a corner

        def analyze_wall_depth(depths):
            """Analyze wall depth using first N points from top"""
            if len(depths) < initial_points:
                return None
                
            # Take first N points and their average depth
            wall_depths = depths[:initial_points]
            mean_depth = np.mean(wall_depths)
            std_depth = np.std(wall_depths)
            
            # Continue adding points while they're within threshold
            for i in range(initial_points, len(depths)):
                if abs(depths[i] - mean_depth) > (3 * std_depth):  # Using 3-sigma rule
                    break
                wall_depths = np.append(wall_depths, depths[i])
            
            return np.mean(wall_depths) if len(wall_depths) >= min_points else None

        def fit_quadratic(x, y):
            """Fit quadratic curve to points: y = ax² + bx + c"""
            A = np.vstack([x**2, x, np.ones(len(x))]).T
            try:
                a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
                return lambda x_new: a * x_new**2 + b * x_new + c, (a, b, c)
            except:
                return None, None

        def analyze_corner_point(depths, x_coords, max_depth_idx, window_size=5):
            """Verify corner point by checking depth consistency of adjacent points"""
            if max_depth_idx < window_size or max_depth_idx >= len(depths) - window_size:
                return False
                
            corner_depth = depths[max_depth_idx]
            
            # Check left side consistency
            left_depths = depths[max_depth_idx-window_size:max_depth_idx]
            left_diff = np.abs(left_depths - corner_depth) / corner_depth
            if not np.all(left_diff < 0.15):  # 15% threshold
                return False
                
            # Check right side consistency
            right_depths = depths[max_depth_idx+1:max_depth_idx+window_size+1]
            right_diff = np.abs(right_depths - corner_depth) / corner_depth
            if not np.all(right_diff < 0.15):  # 15% threshold
                return False
                
            return True

        def analyze_floor_slope(depths, y_coords):
            """Analyze floor using quadratic regression, validating with decreasing slopes"""
            if len(depths) < 30:  # Need enough points for at least 3 groups
                return None, None

            # Reverse arrays to work from bottom up
            depths = depths[::-1]
            y_coords = y_coords[::-1]

            # Calculate average slopes for groups of 10 points
            group_size = 10
            slopes = []

            for i in range(0, len(depths) - group_size, group_size):
                group1_depths = depths[i:i + group_size]
                group1_y = y_coords[i:i + group_size]

                # Calculate average slope for this group using linear regression
                try:
                    slope, _ = np.polyfit(group1_depths, group1_y, 1)
                    slopes.append((slope, i))
                except:
                    continue

            if len(slopes) < 3:  # Need at least 3 groups
                return None, None

            # Find all sequences of decreasing slopes
            sequences = []
            current_sequence = []

            for i in range(len(slopes)):
                if not current_sequence:
                    current_sequence.append(slopes[i])
                elif slopes[i][0] > current_sequence[-1][0]:  # Slope is increasing (becoming less negative)
                    current_sequence.append(slopes[i])
                else:  # Slope increased, start new sequence
                    if len(current_sequence) >= 3:  # Only keep sequences with at least 3 groups
                        sequences.append(current_sequence.copy())
                    current_sequence = [slopes[i]]

            # Don't forget to check the last sequence
            if len(current_sequence) >= 3:
                sequences.append(current_sequence)

            if not sequences:  # No valid sequences found
                return None, None

            # Find the longest sequence
            longest_sequence = max(sequences, key=len)

            if len(longest_sequence) < 3:  # Double check we have enough groups
                return None, None

            # Get the range of points covered by the longest sequence
            start_idx = longest_sequence[0][1]
            end_idx = longest_sequence[-1][1] + group_size

            # Fit quadratic to all points in the valid sequence
            sequence_depths = depths[start_idx:end_idx]
            sequence_y = y_coords[start_idx:end_idx]

            points = np.array(list(zip(sequence_depths, sequence_y)))
            quad_func, params = fit_quadratic(points[:, 0], points[:, 1])

            if quad_func is None:
                return None, None

            return quad_func, params

        def analyze_edge_slope(depths, x_coords, from_left=True):
            """Analyze wall slope at frame edges"""
            if len(depths) < initial_points:
                return None
                
            # Take points from edge region
            if from_left:
                edge_depths = depths[:edge_width]
                edge_x = x_coords[:edge_width]
            else:
                edge_depths = depths[-edge_width:]
                edge_x = x_coords[-edge_width:]
                
            # Fit line to edge points
            try:
                slope, _ = np.polyfit(edge_x, edge_depths, 1)
                return slope
            except:
                return None

        # Storage for intersection points
        floor_wall_points = []

        # First, analyze wall slopes at edges of frame
        has_corner = False
        for y in range(0, depth_map.shape[0] // 2, stripe_width):
            depths = depth_map[y, :]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_x = np.arange(depth_map.shape[1])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[1] * 0.3:
                continue
            
            left_slope = analyze_edge_slope(valid_depths, valid_x, from_left=True)
            right_slope = analyze_edge_slope(valid_depths, valid_x, from_left=False)
            
            if left_slope is not None and right_slope is not None:
                # Check if slopes indicate a corner
                if left_slope * right_slope < 0:  # Slopes in opposite directions
                    has_corner = True
                    break
                elif abs(left_slope - right_slope) > slope_difference_threshold:
                    has_corner = True
                    break

        # Only look for corner if slopes indicate one exists
        corner_x = None
        if has_corner:
            corner_candidates = []
            for y in range(0, depth_map.shape[0] // 2, stripe_width):
                depths = depth_map[y, :]
                valid_mask = depths > 0
                valid_depths = depths[valid_mask]
                valid_x = np.arange(depth_map.shape[1])[valid_mask]
                
                if len(valid_depths) < depth_map.shape[1] * 0.3:
                    continue
                
                # Find the deepest point (corner candidate)
                max_depth_idx = np.argmax(valid_depths)
                if initial_points < max_depth_idx < len(valid_depths) - initial_points:
                    # Verify it's a local maximum
                    if (valid_depths[max_depth_idx] > valid_depths[max_depth_idx-initial_points:max_depth_idx]).all() and \
                       (valid_depths[max_depth_idx] > valid_depths[max_depth_idx+1:max_depth_idx+initial_points+1]).all():
                        corner_x = valid_x[max_depth_idx] * scale_x
                        corner_candidates.append(corner_x)

            # Determine if we have a consistent corner
            if len(corner_candidates) >= min_points:
                median_x = np.median(corner_candidates)
                deviations = np.abs(np.array(corner_candidates) - median_x)
                if np.mean(deviations) < 50:
                    corner_x = median_x

        # Storage for debug visualization
        debug_points = {
            'wall_depths': [],  # (x, depth) pairs
            'floor_points': [],  # (x, y) pairs
            'corner_candidates': [],  # x coordinates
            'floor_wall_points': []  # (x, y) pairs
        }

        # Process vertical stripes for floor-wall intersections
        for x in range(0, depth_map.shape[1], stripe_width):
            depths = depth_map[:, x]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_y = np.arange(depth_map.shape[0])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[0] * 0.3:
                continue

            # Find wall depth starting from top of frame
            wall_depth = analyze_wall_depth(valid_depths)
            
            if wall_depth is not None:
                # Find floor curve starting from bottom of frame
                floor_func, floor_params = analyze_floor_slope(valid_depths, valid_y)
                
                if floor_func is not None:
                    # Get intersection point by directly evaluating quadratic at wall depth
                    try:
                        # The quadratic function already maps depth -> y
                        intersect_y = floor_func(wall_depth)
                        
                        # Verify the intersection point is reasonable
                        if not np.isnan(intersect_y) and 0 <= intersect_y <= frame_height:
                            frame_x = x * scale_x
                            frame_y = intersect_y * scale_y
                            floor_wall_points.append((frame_x, frame_y))
                            
                            # Add to debug visualization
                            # Show floor points up to intersection
                            vis_depths = np.linspace(valid_depths.min(), wall_depth, 20)
                            vis_y = floor_func(vis_depths)
                            debug_points['floor_points'].extend([
                                (x, y) for y in vis_y if not np.isnan(y) and y <= intersect_y
                            ])
                            
                            # Add intersection point to debug
                            debug_points['floor_wall_points'].append((x, intersect_y))
                    except:
                        continue

        # Convert points to lines
        intersection_lines = []

        # Add wall-wall intersection line if corner was found
        if corner_x is not None:
            intersection_lines.append(("wall", ((corner_x, 0), (corner_x, frame_height))))

        # Process floor-wall points
        if len(floor_wall_points) >= min_points:
            points = np.array(floor_wall_points)
            from sklearn.cluster import DBSCAN
            
            # Adjust clustering based on whether we found a corner
            eps = 50 if corner_x is None else 30
            clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
            unique_labels = np.unique(clustering.labels_[clustering.labels_ >= 0])
            
            if corner_x is not None:
                # Split points into left and right of corner
                left_points = points[points[:, 0] < corner_x]
                right_points = points[points[:, 0] >= corner_x]
                
                # Create lines for each side if enough points
                if len(left_points) >= min_points:
                    sorted_points = left_points[left_points[:, 0].argsort()]
                    p1 = tuple(sorted_points[0])
                    p2 = tuple(sorted_points[-1])
                    intersection_lines.append(("floor", (p1, p2)))
                    
                if len(right_points) >= min_points:
                    sorted_points = right_points[right_points[:, 0].argsort()]
                    p1 = tuple(sorted_points[0])
                    p2 = tuple(sorted_points[-1])
                    intersection_lines.append(("floor", (p1, p2)))
            else:
                # Single wall case - one floor line
                for label in unique_labels:
                    cluster_points = points[clustering.labels_ == label]
                    if len(cluster_points) >= min_points:
                        sorted_points = cluster_points[cluster_points[:, 0].argsort()]
                        p1 = tuple(sorted_points[0])
                        p2 = tuple(sorted_points[-1])
                        intersection_lines.append(("floor", (p1, p2)))



        # Process vertical stripes
        for x in range(0, depth_map.shape[1], stripe_width):
            depths = depth_map[:, x]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_y = np.arange(depth_map.shape[0])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[0] * 0.3:
                continue

            # Find wall depth
            wall_depth = analyze_wall_depth(valid_depths)
            if wall_depth is not None:
                debug_points['wall_depths'].append((x, wall_depth))

            # Find floor curve
            floor_func, floor_params = analyze_floor_slope(valid_depths, valid_y)
            if floor_func is not None:
                # Store floor points for visualization
                test_depths = np.linspace(valid_depths.min(), valid_depths.max(), 20)
                floor_y_values = floor_func(test_depths)
                debug_points['floor_points'].extend(zip([x]*len(test_depths), floor_y_values))

        # Process horizontal stripes for corner detection
        corner_candidates = []
        for y in range(0, depth_map.shape[0] // 2, stripe_width):
            depths = depth_map[y, :]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_x = np.arange(depth_map.shape[1])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[1] * 0.3:
                continue
            
            # Find the deepest point
            max_depth_idx = np.argmax(valid_depths)
            
            # Verify corner point consistency
            if analyze_corner_point(valid_depths, valid_x, max_depth_idx):
                corner_candidates.append((valid_x[max_depth_idx], y))

        # Draw debug visualization
        # Draw wall depths
        for x, depth in debug_points['wall_depths']:
            try:
                cv2.circle(debug_frame, (int(x), int(depth)), 2, (0, 255, 0), -1)  # Green
            except:
                continue

        # Draw floor points
        for x, y in debug_points['floor_points']:
            try:
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(debug_frame, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue
            except:
                continue

        # Draw corner candidates in red
        for x, y in corner_candidates:
            try:
                cv2.circle(debug_frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
            except:
                continue

        # Draw floor-wall intersection points in cyan
        for x, y in debug_points['floor_wall_points']:
            try:
                cv2.circle(debug_frame, (int(x), int(y)), 4, (255, 255, 0), -1)  # Cyan (255, 255, 0)
            except:
                continue

        # Draw final intersection lines
        for line_type, (p1, p2) in intersection_lines:
            try:
                p1_depth = (int(p1[0] / scale_x), int(p1[1] / scale_y))
                p2_depth = (int(p2[0] / scale_x), int(p2[1] / scale_y))
                color = (0, 255, 255) if line_type == "wall" else (255, 255, 0)  # Cyan for wall, Yellow for floor
                cv2.line(debug_frame, p1_depth, p2_depth, color, 2)
            except:
                continue

        # Add text labels for clarity
        cv2.putText(debug_frame, "Wall depths (green)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_frame, "Floor points (blue)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(debug_frame, "Corner candidates (red)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(debug_frame, "Floor-wall intersections (cyan)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Show debug visualization
        cv2.imshow('Depth Analysis Debug', debug_frame)
        cv2.waitKey(1)

        return intersection_lines
