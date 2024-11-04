import numpy as np
import open3d as o3d
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

    def depth_map_to_point_cloud(self, depth_map, focal_length, frame_width, frame_height, downsample_factor=30):
        """Convert depth map to point cloud using ray casting, preserving depth values"""
        h, w = depth_map.shape
        points = []
        depths = []  # Store original depth values
        
        # Calculate scaling factors to map depth map coordinates to frame coordinates
        scale_x = frame_width / w
        scale_y = frame_height / h
        
        # Create sampling grid based on downsample factor
        step = int(np.sqrt(downsample_factor))
        for v in range(0, h, step):
            for u in range(0, w, step):
                depth = depth_map[v, u]
                
                # Skip invalid depths
                if depth <= 0 or depth > 30.0:  # Max 30 meters
                    continue
                    
                # Scale depth map coordinates to frame coordinates
                frame_x = u * scale_x
                frame_y = v * scale_y
                
                # Create ray from camera through this pixel
                ray = self.project_point_to_world(
                    np.array([frame_x, frame_y]), 
                    np.array([0, 0, 0, 1]),  # Identity rotation (camera space)
                    focal_length,
                    frame_width,
                    frame_height
                )
                
                # Calculate 3D point by extending ray by depth distance
                point = ray * depth
                points.append(point)
                depths.append(depth)  # Store the depth
        
        # Convert to numpy arrays
        points = np.array(points)
        depths = np.array(depths)
        
        return points, depths

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

    @staticmethod
    def estimate_planes(points, distance_threshold=0.15, ransac_n=3, num_iterations=2000):
        """Estimate planes with tighter RANSAC parameters"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals with tighter radius
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        
        plane_models = []
        planes_inliers = []
        remaining_points = pcd
        min_points = max(100, len(points) // 20)
        
        for _ in range(8):  # Increased max planes to 8 to catch more distinct surfaces
            if len(remaining_points.points) < min_points:
                break
                
            plane_model, inliers = remaining_points.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            if len(inliers) < min_points:
                break
                
            if plane_model[2] < 0:
                plane_model = -np.array(plane_model)
                
            plane_models.append(plane_model)
            planes_inliers.append(inliers)
            
            remaining_points = remaining_points.select_by_index(inliers, invert=True)
        
        return plane_models, planes_inliers

    @staticmethod
    def identify_planes(plane_models):
        """Identify planes with improved normal checks for room-scale geometry"""
        floor_planes = []
        wall_planes = []
        
        for plane_model in plane_models:
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            d = plane_model[3]
            
            # Get angles with coordinate axes
            angle_y = np.abs(np.dot(normal, [0, 1, 0]))  # Angle with Y axis
            angle_z = np.abs(np.dot(normal, [0, 0, 1]))  # Angle with Z axis
            angle_x = np.abs(np.dot(normal, [1, 0, 0]))  # Angle with X axis
            
            # Floor/ceiling: strong Y component and reasonable height
            if angle_y > 0.9:
                # Check if the plane is at a reasonable height (d/normal[1] gives height)
                height = -d/normal[1]
                if -5 < height < 5:  # Allow for both floor and ceiling up to 5m
                    floor_planes.append(plane_model)
            # Walls: strong X or Z component, weak Y component
            elif angle_y < 0.3 and (angle_x > 0.7 or angle_z > 0.7):
                wall_planes.append(plane_model)
        
        return floor_planes, wall_planes

    @staticmethod
    def compute_intersections(floor_planes, wall_planes):
        """Compute both floor-wall and wall-wall intersections"""
        intersection_lines = []
        
        # Floor-wall intersections
        for floor_plane in floor_planes:
            for wall_plane in wall_planes:
                n1 = np.array(floor_plane[:3])
                n2 = np.array(wall_plane[:3])
                
                # Normalize normals
                n1 = n1 / np.linalg.norm(n1)
                n2 = n2 / np.linalg.norm(n2)
                
                # Compute line direction
                direction = np.cross(n1, n2)
                
                # Skip if direction is invalid
                if np.linalg.norm(direction) < 1e-6:
                    continue
                    
                direction = direction / np.linalg.norm(direction)
                
                # Skip if line is too vertical (for floor-wall intersections)
                if abs(direction[1]) > 0.1:
                    continue
                    
                # Solve for point on line
                A = np.vstack([n1, n2])
                b = -np.array([floor_plane[3], wall_plane[3]])
                
                try:
                    point_on_line = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                    # Skip if point is unreasonably far from origin
                    if np.linalg.norm(point_on_line) > 30.0:
                        continue
                        
                    # Ensure consistent direction (right when viewed from above)
                    if direction[0] < 0:
                        direction = -direction
                        
                    intersection_lines.append((point_on_line, direction, True))  # True for floor-wall
                except np.linalg.LinAlgError:
                    continue
        
        # Wall-wall intersections
        for i in range(len(wall_planes)):
            for j in range(i + 1, len(wall_planes)):
                wall1 = wall_planes[i]
                wall2 = wall_planes[j]
                
                n1 = np.array(wall1[:3])
                n2 = np.array(wall2[:3])
                
                # Normalize normals
                n1 = n1 / np.linalg.norm(n1)
                n2 = n2 / np.linalg.norm(n2)
                
                # Check if walls are nearly parallel
                if abs(np.dot(n1, n2)) > 0.95:  # cos(18°) ≈ 0.95
                    continue
                
                # Compute line direction
                direction = np.cross(n1, n2)
                
                # Skip if direction is invalid
                if np.linalg.norm(direction) < 1e-6:
                    continue
                    
                direction = direction / np.linalg.norm(direction)
                
                # For wall-wall intersections, direction should be mostly vertical
                if abs(direction[1]) < 0.9:  # More than ~25° from vertical
                    continue
                
                # Solve for point on line
                A = np.vstack([n1, n2])
                b = -np.array([wall1[3], wall2[3]])
                
                try:
                    point_on_line = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                    # Skip if point is unreasonably far from origin
                    if np.linalg.norm(point_on_line) > 30.0:
                        continue
                        
                    # Ensure consistent direction (up)
                    if direction[1] < 0:
                        direction = -direction
                        
                    intersection_lines.append((point_on_line, direction, False))  # False for wall-wall
                except np.linalg.LinAlgError:
                    continue
        
        return intersection_lines

    def find_wall_floor_intersections_for_frame(self, frame_num, focal_length, frame_width, frame_height):
        """Find wall-floor intersections using depth line analysis"""
        depth_map = self.load_depth_map(frame_num)
        if depth_map is None:
            return [], (np.zeros((0, 3)), np.array([])), (np.zeros((0, 3)), np.array([]))

        # Step 1: Find the deepest point(s) in the frame
        max_depth = np.max(depth_map[depth_map > 0])  # Ignore zero depths
        deep_points = np.where(depth_map > max_depth * 0.95)  # Points within 5% of max depth
        
        # Calculate centroid of deep region
        deep_center = np.array([np.mean(deep_points[0]), np.mean(deep_points[1])])
        
        # Step 2: Search for vertical corner edge
        corner_found = False
        corner_x = None
        vertical_line_depths = []
        
        # Search around deep center for vertical lines with consistent depth
        search_width = 100  # pixels to search around deep center
        for x in range(max(0, int(deep_center[1] - search_width)), 
                       min(depth_map.shape[1], int(deep_center[1] + search_width))):
            # Get depth values along vertical line
            depths = depth_map[:, x]
            valid_depths = depths[depths > 0]
            
            if len(valid_depths) < depth_map.shape[0] * 0.5:  # Need at least 50% valid depths
                continue
                
            # Calculate depth consistency along line
            depth_std = np.std(valid_depths)
            depth_mean = np.mean(valid_depths)
            
            if depth_std < 0.1 * depth_mean:  # Depth variation less than 10%
                corner_found = True
                corner_x = x
                vertical_line_depths = valid_depths
                break
        
        # Step 3: Generate parallel depth sampling lines
        sample_lines = []
        num_lines = 20  # Number of vertical lines to sample
        
        if corner_found:
            # Use corner as reference
            start_x = max(0, corner_x - 200)
            end_x = min(depth_map.shape[1], corner_x + 200)
        else:
            # Use center of frame as reference
            center_x = depth_map.shape[1] // 2
            start_x = max(0, center_x - 200)
            end_x = min(depth_map.shape[1], center_x + 200)
        
        x_positions = np.linspace(start_x, end_x, num_lines)
        
        # Step 4: Find wall-floor intersections along each line
        intersection_points = []
        
        def ransac_line_fit(depths, y_coords, num_iterations=100):
            """Fit line to depth values using RANSAC"""
            best_score = 0
            best_slope = 0
            best_intercept = 0
            
            for _ in range(num_iterations):
                # Randomly select two points
                idx = np.random.choice(len(depths), 2, replace=False)
                y1, y2 = y_coords[idx]
                d1, d2 = depths[idx]
                
                # Calculate slope and intercept
                if y2 - y1 != 0:
                    slope = (d2 - d1) / (y2 - y1)
                    intercept = d1 - slope * y1
                    
                    # Count inliers
                    predicted = slope * y_coords + intercept
                    inliers = np.abs(predicted - depths) < 0.1  # 10cm threshold
                    score = np.sum(inliers)
                    
                    if score > best_score:
                        best_score = score
                        best_slope = slope
                        best_intercept = intercept
            
            return best_slope, best_intercept
        
        for x in x_positions:
            x_int = int(x)
            depths = depth_map[:, x_int]
            valid_mask = depths > 0
            valid_depths = depths[valid_mask]
            valid_y = np.arange(depth_map.shape[0])[valid_mask]
            
            if len(valid_depths) < depth_map.shape[0] * 0.5:
                continue
            
            # Fit lines to top and bottom portions
            mid_idx = len(valid_depths) // 2
            
            # Bottom half (floor)
            floor_slope, floor_intercept = ransac_line_fit(
                valid_depths[:mid_idx], valid_y[:mid_idx])
            
            # Top half (wall)
            wall_slope, wall_intercept = ransac_line_fit(
                valid_depths[mid_idx:], valid_y[mid_idx:])
            
            # Find intersection
            if abs(wall_slope - floor_slope) > 1e-6:
                y_intersect = (floor_intercept - wall_intercept) / (wall_slope - floor_slope)
                if 0 <= y_intersect < depth_map.shape[0]:
                    # Calculate depth at intersection
                    depth_intersect = wall_slope * y_intersect + wall_intercept
                    intersection_points.append((x_int, int(y_intersect), depth_intersect))
        
        # Step 5: Convert intersection points to lines
        intersection_lines = []
        if len(intersection_points) >= 2:
            # Sort points by x coordinate
            points = np.array(intersection_points)
            sorted_indices = np.argsort(points[:, 0])
            points = points[sorted_indices]

            # Group points into left and right sections
            mid_x = (points[0, 0] + points[-1, 0]) / 2
            left_points = points[points[:, 0] < mid_x]
            right_points = points[points[:, 0] >= mid_x]

            # Fit lines to each section if enough points
            def fit_line_to_points(pts):
                if len(pts) < 2:
                    return None
                # Use first and last point to define line
                p1 = pts[0]
                p2 = pts[-1]
                point = np.array([p1[0], p1[1], p1[2]])  # x, y, depth
                direction = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
                direction = direction / np.linalg.norm(direction)
                return point, direction

            # Create floor-wall intersection lines
            if len(left_points) >= 2:
                left_line = fit_line_to_points(left_points)
                if left_line:
                    point, direction = left_line
                    intersection_lines.append((point, direction, True))  # True for floor-wall

            if len(right_points) >= 2:
                right_line = fit_line_to_points(right_points)
                if right_line:
                    point, direction = right_line
                    intersection_lines.append((point, direction, True))

            # Add vertical wall-wall intersections at corners if we found both lines
            if len(intersection_lines) == 2:
                # Use the intersection of the two floor lines to define the corner
                p1, d1, _ = intersection_lines[0]
                p2, d2, _ = intersection_lines[1]
                
                # Find intersection point of the two lines
                # Use the first points of both lines to define the vertical line
                corner_point = (p1 + p2) / 2
                vertical_direction = np.array([0, 1, 0])  # Vertical direction
                
                # Add vertical wall-wall intersection
                intersection_lines.append((corner_point, vertical_direction, False))  # False for wall-wall

        # Create empty point clouds for compatibility
        empty_points = np.zeros((0, 3))
        empty_depths = np.array([])

        return intersection_lines, (empty_points, empty_depths), (empty_points, empty_depths)
