import numpy as np
import open3d as o3d
import os

class SurfaceDetector:
    def __init__(self, depth_dir, pixel_to_meter=0.000264583):
        self.depth_dir = depth_dir
        self.cx = 320  # pixel coordinates
        self.cy = 240  # pixel coordinates
        self.pixel_to_meter = pixel_to_meter  # conversion factor from pixels to meters

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

    def depth_map_to_point_cloud(self, depth_map, focal_length, downsample_factor=100):
        h, w = depth_map.shape
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        z = depth_map.flatten()
        valid = z > 0
        u = u[valid]
        v = v[valid]
        z = z[valid]

        # Convert pixel coordinates to meters before computing 3D coordinates
        x = ((u - self.cx) * self.pixel_to_meter) * z / focal_length
        y = ((v - self.cy) * self.pixel_to_meter) * z / focal_length
        points = np.vstack((x, y, z)).T

        # Downsample the points by a factor of 100
        if downsample_factor > 1:
            indices = np.random.choice(points.shape[0], points.shape[0] // downsample_factor, replace=False)
            points = points[indices]

        return points

    @staticmethod
    def estimate_planes(points, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane_models = []
        planes_inliers = []
        max_planes = 5  # Adjust based on expected number of planes
        for _ in range(max_planes):
            if len(pcd.points) < 100:
                break
            plane_model, inliers = pcd.segment_plane(distance_threshold,
                                                     ransac_n,
                                                     num_iterations)
            if len(inliers) < 50:
                break
            plane_models.append(plane_model)
            planes_inliers.append(inliers)
            pcd = pcd.select_by_index(inliers, invert=True)
        return plane_models, planes_inliers

    @staticmethod
    def identify_planes(plane_models):
        floor_planes = []
        wall_planes = []
        for plane_model in plane_models:
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            abs_normal_1 = abs(normal[1])
            if abs_normal_1 > 0.9:
                floor_planes.append(plane_model)
            else:
                wall_planes.append(plane_model)
        return floor_planes, wall_planes

    @staticmethod
    def compute_intersections(floor_planes, wall_planes):
        intersection_lines = []
        for floor_plane in floor_planes:
            for wall_plane in wall_planes:
                n1 = np.array(floor_plane[:3])
                n2 = np.array(wall_plane[:3])
                n1 = n1 / np.linalg.norm(n1)
                n2 = n2 / np.linalg.norm(n2)
                direction = np.cross(n1, n2)
                if np.linalg.norm(direction) < 1e-6:
                    continue
                A = np.array([n1, n2])
                b = -np.array([floor_plane[3], wall_plane[3]])
                try:
                    point_on_line = np.linalg.lstsq(A, b, rcond=None)[0]
                    intersection_lines.append((point_on_line, direction))
                except np.linalg.LinAlgError:
                    continue
        return intersection_lines

    def find_wall_floor_intersections(self, depth_map, focal_length):
        points = self.depth_map_to_point_cloud(depth_map, focal_length)
        plane_models, planes_inliers = self.estimate_planes(points)
        floor_planes, wall_planes = self.identify_planes(plane_models)
        intersection_lines = self.compute_intersections(floor_planes, wall_planes)
        return intersection_lines

    def find_wall_floor_intersections_for_frame(self, frame_num, focal_length):
        return self.find_wall_floor_intersections(self.load_depth_map(frame_num), focal_length)
