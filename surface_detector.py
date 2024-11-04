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

        def analyze_floor_slope(depths, y_coords, wall_depth):
            """
            Analyze floor using quadratic regression from bottom up.
            Returns quadratic function and its parameters.
            """
            if len(depths) < initial_points:
                return None, None
                
            # Reverse arrays to work from bottom up
            depths = depths[::-1]
            y_coords = y_coords[::-1]
            
            # Take first N points to establish initial curve
            floor_points = list(zip(depths[:initial_points], y_coords[:initial_points]))
            points = np.array(floor_points)
            quad_func, params = fit_quadratic(points[:, 0], points[:, 1])
            
            if quad_func is None:
                return None, None
                
            # Variables to track best fit
            best_points = floor_points.copy()
            best_func = quad_func
            best_params = params
            consecutive_outliers = 0
            
            # Continue adding points while they follow the curve
            for i in range(initial_points, len(depths)):
                depth = depths[i]
                y = y_coords[i]
                
                predicted_y = quad_func(depth)
                error = abs(y - predicted_y) / abs(predicted_y)
                
                # Stop if we reach wall depth (but don't use this point for intersection)
                if abs(depth - wall_depth) < 0.1:
                    break
                    
                if error > outlier_threshold:
                    consecutive_outliers += 1
                    if consecutive_outliers >= 3:
                        break
                else:
                    consecutive_outliers = 0
                    floor_points.append((depth, y))
                    
                    # Update quadratic fit every few points
                    if len(floor_points) % 3 == 0:
                        points = np.array(floor_points)
                        new_func, new_params = fit_quadratic(points[:, 0], points[:, 1])
                        if new_func is not None:
                            quad_func = new_func
                            params = new_params
                            best_points = floor_points.copy()
                            best_func = quad_func
                            best_params = params
            
            if len(best_points) < min_points:
                return None, None
                
            return best_func, best_params

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
                floor_func, floor_params = analyze_floor_slope(valid_depths, valid_y, wall_depth)
                
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
            floor_func, floor_params = analyze_floor_slope(valid_depths, valid_y, wall_depth)
            if floor_func is not None:
                # Store floor points for visualization
                test_depths = np.linspace(valid_depths.min(), valid_depths.max(), 20)
                floor_y_values = floor_func(test_depths)
                debug_points['floor_points'].extend(zip([x]*len(test_depths), floor_y_values))

        # Process horizontal stripes for corner detection
        for y in range(0, depth_map.shape[0] // 2, stripe_width):
            depths = depth_map[y, :]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_x = np.arange(depth_map.shape[1])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[1] * 0.3:
                continue

            # Find deepest point
            max_depth_idx = np.argmax(valid_depths)
            if initial_points < max_depth_idx < len(valid_depths) - initial_points:
                if (valid_depths[max_depth_idx] > valid_depths[max_depth_idx-initial_points:max_depth_idx]).all() and \
                   (valid_depths[max_depth_idx] > valid_depths[max_depth_idx+1:max_depth_idx+initial_points+1]).all():
                    corner_x = valid_x[max_depth_idx]
                    debug_points['corner_candidates'].append((corner_x, y))

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
                if not np.isnan(x) and not np.isnan(y):  # Check for NaN values
                    cv2.circle(debug_frame, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue
            except:
                continue

        # Draw corner candidates
        for x, y in debug_points['corner_candidates']:
            try:
                cv2.circle(debug_frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red
            except:
                continue

        # Draw floor-wall intersection points
        for x, y in floor_wall_points:
            try:
                x_depth = int(x / scale_x)
                y_depth = int(y / scale_y)
                # Use different colors for natural vs calculated intersections
                color = (0, 255, 255) if natural_intersection else (255, 255, 0)
                cv2.circle(debug_frame, (x_depth, y_depth), 4, color, -1)
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

        # Show debug visualization
        cv2.imshow('Depth Analysis Debug', debug_frame)
        cv2.waitKey(1)

        return intersection_lines
