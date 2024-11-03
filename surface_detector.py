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
        """Estimate planes with improved RANSAC parameters"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals (this helps with plane detection)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        plane_models = []
        planes_inliers = []
        remaining_points = pcd
        min_points = max(100, len(points) // 20)  # At least 5% of points for a plane
        
        for _ in range(5):  # Try to find up to 5 planes
            if len(remaining_points.points) < min_points:
                break
                
            # Segment plane with increased iterations and threshold
            plane_model, inliers = remaining_points.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            if len(inliers) < min_points:
                break
                
            # Ensure plane normal points towards camera
            if plane_model[2] < 0:  # If Z component is negative
                plane_model = -np.array(plane_model)  # Flip normal
                
            plane_models.append(plane_model)
            planes_inliers.append(inliers)
            
            # Remove inliers and continue
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
        depth_map = self.load_depth_map(frame_num)
        points, depths = self.depth_map_to_point_cloud(depth_map, focal_length, frame_width, frame_height)
        plane_models, planes_inliers = self.estimate_planes(points)
        floor_planes, wall_planes = self.identify_planes(plane_models)
        intersection_lines = self.compute_intersections(floor_planes, wall_planes)
        
        # Classify points based on planes using the inliers directly
        floor_points = []
        floor_depths = []
        wall_points = []
        wall_depths = []
        
        # Keep track of used points to avoid duplicates
        used_points = set()
        
        # First, use the inlier indices from RANSAC
        for plane_idx, inliers in enumerate(planes_inliers):
            # Check if this is a floor or wall plane
            plane_model = plane_models[plane_idx]
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            is_floor = abs(normal[1]) > 0.9  # Check Y component for floor
            
            for idx in inliers:
                if idx not in used_points:
                    point = points[idx]
                    depth = depths[idx]
                    if is_floor:
                        floor_points.append(point)
                        floor_depths.append(depth)
                    else:
                        wall_points.append(point)
                        wall_depths.append(depth)
                    used_points.add(idx)
        
        # Convert to numpy arrays for better handling
        floor_points = np.array(floor_points) if floor_points else np.zeros((0, 3))
        floor_depths = np.array(floor_depths) if floor_depths else np.array([])
        wall_points = np.array(wall_points) if wall_points else np.zeros((0, 3))
        wall_depths = np.array(wall_depths) if wall_depths else np.array([])
        
        print(f"Found {len(floor_points)} floor points and {len(wall_points)} wall points")
        
        return intersection_lines, (floor_points, floor_depths), (wall_points, wall_depths)
