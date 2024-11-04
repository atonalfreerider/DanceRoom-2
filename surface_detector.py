import numpy as np
import os
from scipy.spatial.transform import Rotation

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

        def analyze_floor_slope(depths, y_coords):
            """Analyze floor using quadratic regression from bottom up"""
            if len(depths) < initial_points:
                return None, None
                
            # Reverse arrays to work from bottom up
            depths = depths[::-1]
            y_coords = y_coords[::-1]
            
            # Take first N points
            floor_points = list(zip(depths[:initial_points], y_coords[:initial_points]))
            
            # Continue adding points while they follow the curve
            points = np.array(floor_points)
            quad_func, params = fit_quadratic(points[:, 0], points[:, 1])
            
            if quad_func is None:
                return None, None
                
            for i in range(initial_points, len(depths)):
                depth = depths[i]
                y = y_coords[i]
                
                predicted_y = quad_func(depth)
                if abs(y - predicted_y) > (abs(predicted_y) * outlier_threshold):
                    break
                    
                floor_points.append((depth, y))
            
            if len(floor_points) >= min_points:
                points = np.array(floor_points)
                return fit_quadratic(points[:, 0], points[:, 1])
                
            return None, None

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
                    # Find intersection point where floor curve meets wall depth
                    try:
                        y_values = np.linspace(0, frame_height, 100)
                        depths = np.array([wall_depth] * len(y_values))
                        differences = np.abs(floor_func(depths) - y_values)
                        intersect_idx = np.argmin(differences)
                        
                        if differences[intersect_idx] < 10:  # Threshold for valid intersection
                            frame_x = x * scale_x
                            frame_y = y_values[intersect_idx] * scale_y
                            
                            if 0 <= frame_y <= frame_height:
                                floor_wall_points.append((frame_x, frame_y))
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

        return intersection_lines
