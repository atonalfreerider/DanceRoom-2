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

        # Parameters for moving average analysis
        window_size = 5  # Size of moving average window
        outlier_threshold = 0.35  # 15% deviation from moving average is considered an outlier
        consecutive_outliers_threshold = 20  # Number of consecutive outliers to confirm boundary
        min_points = 10  # Minimum points needed for valid wall or floor detection

        def analyze_wall_depth(depths):
            """Analyze wall depth using moving average from top down"""
            if len(depths) < min_points:
                return None
                
            wall_depths = []
            moving_avg = depths[0]  # Initialize with first depth
            consecutive_outliers = 0
            
            for i in range(1, len(depths)):
                depth = depths[i]
                # Check if current depth is an outlier
                if abs(depth - moving_avg) > (moving_avg * outlier_threshold):
                    consecutive_outliers += 1
                    if consecutive_outliers >= consecutive_outliers_threshold:
                        break
                else:
                    consecutive_outliers = 0
                    wall_depths.append(depth)
                    # Update moving average
                    moving_avg = np.mean(wall_depths[-window_size:])
            
            return np.mean(wall_depths) if len(wall_depths) >= min_points else None

        def analyze_floor_slope(depths, y_coords):
            """Analyze floor slope using moving average of local slopes from bottom up"""
            if len(depths) < min_points:
                return None, None
                
            floor_points = []
            slopes = []
            moving_avg_slope = None
            consecutive_outliers = 0
            
            # Start from bottom of frame
            for i in range(len(depths)-2, -1, -1):
                if len(floor_points) < 2:
                    floor_points.append((depths[i], y_coords[i]))
                    continue
                    
                # Calculate local slope
                dy = y_coords[i] - floor_points[-1][1]
                dx = depths[i] - floor_points[-1][0]
                if abs(dx) < 1e-6:  # Avoid division by zero
                    continue
                    
                current_slope = dy / dx
                
                if moving_avg_slope is None:
                    moving_avg_slope = current_slope
                    slopes.append(current_slope)
                    floor_points.append((depths[i], y_coords[i]))
                else:
                    # Check if current slope is an outlier
                    if abs(current_slope - moving_avg_slope) > (abs(moving_avg_slope) * outlier_threshold):
                        consecutive_outliers += 1
                        if consecutive_outliers >= consecutive_outliers_threshold:
                            break
                    else:
                        consecutive_outliers = 0
                        slopes.append(current_slope)
                        floor_points.append((depths[i], y_coords[i]))
                        # Update moving average slope
                        moving_avg_slope = np.mean(slopes[-window_size:])
            
            if len(floor_points) >= min_points:
                floor_points = np.array(floor_points)
                final_slope = np.mean(slopes)
                # Calculate intercept using middle point
                mid_point = floor_points[len(floor_points)//2]
                intercept = mid_point[1] - final_slope * mid_point[0]
                return final_slope, intercept
                
            return None, None

        def analyze_wall_slope(depths, x_coords, from_left=True):
            """Analyze wall slope using moving average from either left or right side"""
            if len(depths) < min_points:
                return None, None
                
            wall_points = []
            slopes = []
            moving_avg_slope = None
            consecutive_outliers = 0
            
            # Determine iteration direction
            if from_left:
                range_iter = range(1, len(depths))
            else:
                range_iter = range(len(depths)-2, -1, -1)
                
            for i in range_iter:
                if len(wall_points) < 2:
                    wall_points.append((x_coords[i], depths[i]))
                    continue
                    
                # Calculate local slope
                dx = x_coords[i] - wall_points[-1][0]
                dz = depths[i] - wall_points[-1][1]  # Using z for depth
                if abs(dx) < 1e-6:  # Avoid division by zero
                    continue
                    
                current_slope = dz / dx
                
                if moving_avg_slope is None:
                    moving_avg_slope = current_slope
                    slopes.append(current_slope)
                    wall_points.append((x_coords[i], depths[i]))
                else:
                    # Check if current slope is an outlier
                    if abs(current_slope - moving_avg_slope) > (abs(moving_avg_slope) * outlier_threshold):
                        consecutive_outliers += 1
                        if consecutive_outliers >= consecutive_outliers_threshold:
                            break
                    else:
                        consecutive_outliers = 0
                        slopes.append(current_slope)
                        wall_points.append((x_coords[i], depths[i]))
                        # Update moving average slope
                        moving_avg_slope = np.mean(slopes[-window_size:])
            
            if len(wall_points) >= min_points:
                wall_points = np.array(wall_points)
                final_slope = np.mean(slopes)
                # Calculate intercept using middle point
                mid_point = wall_points[len(wall_points)//2]
                intercept = mid_point[1] - final_slope * mid_point[0]
                return final_slope, intercept
                
            return None, None

        # Storage for intersection points
        floor_wall_points = []

        # Process vertical stripes
        stripe_width = 5
        for x in range(0, depth_map.shape[1], stripe_width):
            depths = depth_map[:, x]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_y = np.arange(depth_map.shape[0])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[0] * 0.3:  # Skip if too few valid points
                continue

            # Find wall depth starting from top of frame
            wall_depth = analyze_wall_depth(valid_depths)
            
            if wall_depth is not None:
                # Find floor slope starting from bottom of frame
                floor_slope, floor_intercept = analyze_floor_slope(valid_depths, valid_y)
                
                if floor_slope is not None:
                    # Find intersection point
                    # y = mx + b -> solve for y where x = wall_depth
                    intersect_y = floor_slope * wall_depth + floor_intercept
                    
                    # Convert to frame coordinates
                    frame_x = x * scale_x
                    frame_y = intersect_y * scale_y
                    
                    if 0 <= frame_y <= frame_height:
                        floor_wall_points.append((frame_x, frame_y))

        # Process horizontal stripes in top half for wall-wall intersection
        wall_intersections = []
        
        for y in range(0, depth_map.shape[0] // 2, stripe_width):
            depths = depth_map[y, :]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_x = np.arange(depth_map.shape[1])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[1] * 0.3:
                continue
                
            # Analyze wall slopes from both sides
            left_slope, left_intercept = analyze_wall_slope(valid_depths, valid_x, from_left=True)
            right_slope, right_intercept = analyze_wall_slope(valid_depths, valid_x, from_left=False)
            
            if left_slope is not None and right_slope is not None:
                # Check if slopes are significantly different (indicating a corner)
                slope_difference = abs(left_slope - right_slope)
                if slope_difference > 0.1:  # Threshold for different slopes
                    # Calculate intersection point
                    # z1 = m1*x + b1, z2 = m2*x + b2
                    # At intersection: m1*x + b1 = m2*x + b2
                    # x = (b2 - b1)/(m1 - m2)
                    try:
                        intersect_x = (left_intercept - right_intercept) / (right_slope - left_slope)
                        if 0 <= intersect_x <= frame_width:
                            wall_intersections.append(intersect_x)
                    except ZeroDivisionError:
                        continue
                else:
                    # Slopes are similar - this is likely a single wall
                    continue

        # Convert points to lines
        intersection_lines = []

        # If we have consistent wall intersections, add a vertical line
        if len(wall_intersections) >= min_points:
            # Use median x position for stability
            corner_x = np.median(wall_intersections)
            # Only add if we have floor-wall points (for consistency check)
            if len(floor_wall_points) >= min_points:
                # Verify corner position is between floor-wall points
                min_x = min(p[0] for p in floor_wall_points)
                max_x = max(p[0] for p in floor_wall_points)
                if min_x <= corner_x <= max_x:
                    intersection_lines.append(("wall", ((corner_x, 0), (corner_x, frame_height))))
        
        # Process floor-wall points using RANSAC
        if len(floor_wall_points) >= min_points:
            points = np.array(floor_wall_points)
            # Use RANSAC to fit line(s) to floor-wall points
            from sklearn.cluster import DBSCAN
            
            # Try to identify if we have one or two lines using clustering
            clustering = DBSCAN(eps=50, min_samples=min_points).fit(points)
            unique_labels = np.unique(clustering.labels_[clustering.labels_ >= 0])
            
            for label in unique_labels:
                cluster_points = points[clustering.labels_ == label]
                if len(cluster_points) >= min_points:
                    # Sort points by x coordinate
                    sorted_points = cluster_points[cluster_points[:, 0].argsort()]
                    p1 = tuple(sorted_points[0])
                    p2 = tuple(sorted_points[-1])
                    intersection_lines.append(("floor", (p1, p2)))

        return intersection_lines
