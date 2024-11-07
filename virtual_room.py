import numpy as np
import cv2
from scipy.spatial.transform import Rotation


class VirtualRoom:
    def __init__(self):
        # Room dimensions in meters
        # Elevation Crown Plaza Denver Ballroom
        self.room_width = 18.29
        self.room_depth = 11.0
        self.room_height = 4.57
        self.grid_size = 1.0  # 1m^2 grid tiles
        self.frame_width, self.frame_height = None, None

    def set_frame(self, frame_height, frame_width):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def project_point_to_planes(self, world_ray, camera_position):
        # Test intersections
        hw, hd = self.room_width / 2, self.room_depth / 2
        intersections = []

        # Test floor (y = 0)
        if abs(world_ray[1]) > 1e-6:
            t = -camera_position[1] / world_ray[1]
            if t > 0:
                point = camera_position + t * world_ray
                if abs(point[0]) <= hw and abs(point[2]) <= hd:
                    intersections.append(('floor', point, t))

        # Test back wall (z = -hd)
        if abs(world_ray[2]) > 1e-6:
            t = (-hd - camera_position[2]) / world_ray[2]
            if t > 0:
                point = camera_position + t * world_ray
                if abs(point[0]) <= hw and 0 <= point[1] <= self.room_height:
                    intersections.append(('back_wall', point, t))

        # Test left wall (x = -hw)
        if abs(world_ray[0]) > 1e-6:
            t = (-hw - camera_position[0]) / world_ray[0]
            if t > 0:
                point = camera_position + t * world_ray
                if abs(point[2]) <= hd and 0 <= point[1] <= self.room_height:
                    intersections.append(('left_wall', point, t))

        # Test right wall (x = hw)
        if abs(world_ray[0]) > 1e-6:
            t = (hw - camera_position[0]) / world_ray[0]
            if t > 0:
                point = camera_position + t * world_ray
                if abs(point[2]) <= hd and 0 <= point[1] <= self.room_height:
                    intersections.append(('right_wall', point, t))

        # Return closest intersection point and its type
        if intersections:
            intersections.sort(key=lambda x: x[2])  # Sort by distance
            return intersections[0]  # Return the point

        return None

    def project_points(self, vertices, rotation, position, focal_length):
        fx = fy = min(self.frame_height, self.frame_width)
        cx, cy = self.frame_width / 2, self.frame_height / 2

        # Project vertices to image plane
        image_points = []
        behind_camera = []
        for vertex in vertices:
            # Transform point to camera space
            point_cam = rotation.inv().apply(vertex - position)

            # Check if point is behind camera
            if point_cam[2] <= 0:
                behind_camera.append(True)
                image_points.append(None)
                continue

            # Project to image plane with correct perspective
            x = -point_cam[0] / point_cam[2]  # Negative x for correct left-right orientation
            y = -point_cam[1] / point_cam[2]  # Negative y for correct up-down orientation

            # Convert to pixel coordinates with focal length
            pixel_x = int(x * fx * focal_length + cx)
            pixel_y = int(y * fy * focal_length + cy)

            behind_camera.append(False)
            image_points.append((pixel_x, pixel_y))

        return image_points, behind_camera

    def draw_virtual_room(self, frame, position, rotation, focal_length):
        """Draw the virtual room borders and floor origin"""
        # Get camera parameters
        fx = fy = min(self.frame_height, self.frame_width)
        cx, cy = self.frame_width / 2, self.frame_height / 2

        # Generate room vertices with origin at floor center
        hw, hd = self.room_width / 2, self.room_depth / 2
        h = self.room_height  # Full height, since y=0 is now at floor level
        vertices = np.array([
            [-hw, 0, hd], [hw, 0, hd],  # Floor front (closer to camera)
            [-hw, 0, -hd], [hw, 0, -hd],  # Floor back (near back wall)
            [-hw, h, hd], [hw, h, hd],  # Wall top front
            [-hw, h, -hd], [hw, h, -hd]  # Wall top back
        ])

        image_points, behind_camera = self.project_points(
            vertices,
            rotation,
            position,
            focal_length)

        def is_point_in_frame(point):
            if point is None:
                return False
            return (0 <= point[0] < self.frame_width and
                    0 <= point[1] < self.frame_height)

        def clip_line(p1, p2, i1, i2):
            """Clip line to frame boundaries and check visibility"""
            # If either point is behind camera, don't draw
            if behind_camera[i1] or behind_camera[i2]:
                return False

            # If both points are outside frame in same direction, don't draw
            if p1 is not None and p2 is not None:
                if ((p1[0] < 0 and p2[0] < 0) or
                        (p1[0] >= self.frame_width and p2[0] >= self.frame_width) or
                        (p1[1] < 0 and p2[1] < 0) or
                        (p1[1] >= self.frame_height and p2[1] >= self.frame_height)):
                    return False

                # If at least one point is in frame, draw the line
                return True
            return False

        # Draw double lines for floor edges (red)
        floor_edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for edge in floor_edges:
            try:
                p1, p2 = image_points[edge[0]], image_points[edge[1]]
                if clip_line(p1, p2, edge[0], edge[1]):
                    cv2.line(frame, p1, p2, (0, 0, 255), 3)  # Thick red line
                    cv2.line(frame, p1, p2, (128, 0, 128), 1)  # Thin purple outline
            except:
                continue

        # Draw double lines for wall edges (green)
        wall_edges = [
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]
        for edge in wall_edges:
            try:
                p1, p2 = image_points[edge[0]], image_points[edge[1]]
                if clip_line(p1, p2, edge[0], edge[1]):
                    cv2.line(frame, p1, p2, (0, 255, 0), 3)  # Thick green line
                    cv2.line(frame, p1, p2, (0, 128, 128), 1)  # Thin teal outline
            except:
                continue

        # Draw floor grid crosses
        cross_size = 0.2  # 0.2 meters leg length
        # Create evenly spaced grid points in world coordinates
        x_count = int(self.room_width / self.grid_size) - 1
        z_count = int(self.room_depth / self.grid_size) - 1

        # Create grid starting from back to front, left to right
        for i in range(z_count):
            z = -hd + self.grid_size + i * self.grid_size  # Start from back wall
            for j in range(x_count):
                x = hw - self.grid_size - j * self.grid_size  # Start from right wall (flipped x)

                # Define cross points in world coordinates (y=0 for floor)
                cross_center = np.array([x, 0, z])
                cross_points = [
                    cross_center + np.array([-cross_size, 0, 0]),  # Right (flipped)
                    cross_center - np.array([-cross_size, 0, 0]),  # Left (flipped)
                    cross_center + np.array([0, 0, cross_size]),  # Forward
                    cross_center - np.array([0, 0, cross_size])  # Back
                ]

                # Project all points of the cross
                pixel_points = []
                all_visible = True
                for point in [cross_center] + cross_points:
                    # Transform point to camera space
                    point_cam = rotation.inv().apply(point - position)

                    if point_cam[2] <= 0:  # Behind camera
                        all_visible = False
                        break

                    # Project to image plane
                    x_proj = -point_cam[0] / point_cam[2]
                    y_proj = -point_cam[1] / point_cam[2]

                    # Convert to pixel coordinates
                    pixel_x = int(x_proj * fx * focal_length + cx)
                    pixel_y = int(y_proj * fx * focal_length + cy)

                    # Check if point is in frame
                    if not (0 <= pixel_x < self.frame_width and 0 <= pixel_y < self.frame_height):
                        all_visible = False
                        break

                    pixel_points.append((pixel_x, pixel_y))

                # Draw cross if all points are visible
                if all_visible:
                    center = pixel_points[0]
                    right = pixel_points[1]
                    left = pixel_points[2]
                    forward = pixel_points[3]
                    back = pixel_points[4]

                    cv2.line(frame, center, right, (0, 0, 255), 1)  # Red cross
                    cv2.line(frame, center, left, (0, 0, 255), 1)  # Red cross
                    cv2.line(frame, center, forward, (0, 0, 255), 1)  # Red cross
                    cv2.line(frame, center, back, (0, 0, 255), 1)  # Red cross

        # Draw floor origin and axes only if in front of camera
        origin_cam = rotation.inv().apply(-position)
        if origin_cam[2] > 0:  # Only draw if origin is in front of camera
            origin = [0, 0, 0]  # Floor center (y=0)
            origin_cam = rotation.inv().apply(origin - position)

            if origin_cam[2] > 0:
                x = -origin_cam[0] / origin_cam[2]
                y = -origin_cam[1] / origin_cam[2]
                origin_pixel = (int(x * fx * focal_length + cx),
                                int(y * fx * focal_length + cy))

                if is_point_in_frame(origin_pixel):
                    # Draw axes only if origin is visible
                    for axis, color in [
                        ([1, 0, 0], (0, 0, 255)),  # X axis (red)
                        ([0, 1, 0], (0, 255, 0)),  # Y axis (green)
                        ([0, 0, 1], (255, 0, 0))  # Z axis (blue)
                    ]:
                        end_cam = rotation.inv().apply(axis - position)
                        if end_cam[2] > 0:
                            x = -end_cam[0] / end_cam[2]
                            y = -end_cam[1] / end_cam[2]
                            end_pixel = (int(x * fx * focal_length + cx),
                                         int(y * fx * focal_length + cy))

                            if clip_line(origin_pixel, end_pixel, 0, 0):  # Using 0,0 as dummy indices
                                cv2.line(frame, origin_pixel, end_pixel, color, 2)

                    cv2.circle(frame, origin_pixel, 5, (255, 255, 255), -1)  # Origin point in white

        return frame

    def mouse_callback(self, x, y, frame_height, frame_width, position, rotation, focal_length):
        # Debug ray casting
        print("\nRay Debug:")
        print(f"Clicked pixel: ({x}, {y})")

        # Get camera parameters
        fx = fy = min(frame_height, frame_width)
        cx, cy = frame_width / 2, frame_height / 2
        rotation = Rotation.from_quat(rotation)

        # Convert to normalized device coordinates
        x_ndc = -(x - cx) / fx  # Flip X to match world space
        y_ndc = (cy - y) / fy  # Flip Y to match world space

        # Create ray in camera space (positive Z is forward)
        ray_dir = np.array([
            x_ndc / focal_length,
            y_ndc / focal_length,
            1.0  # Changed to positive Z for forward
        ])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        # Transform ray to world space
        world_ray = rotation.apply(ray_dir)
        print(f"Ray direction (world space): {world_ray}")

        # Test intersections
        hw, hd = self.room_width / 2, self.room_depth / 2

        # Test floor (y = 0)
        if abs(world_ray[1]) > 1e-6:
            t = -position[1] / world_ray[1]
            if t > 0:
                point = position + t * world_ray
                if abs(point[0]) <= hw and abs(point[2]) <= hd:
                    print(f"Floor intersection at: {point}")

        # Test back wall (z = -hd)
        if abs(world_ray[2]) > 1e-6:
            t = (-hd - position[2]) / world_ray[2]
            if t > 0:
                point = position + t * world_ray
                if abs(point[0]) <= hw and 0 <= point[1] <= self.room_height:
                    print(f"Back wall intersection at: {point}")

        # Test left wall (x = hw)  # Flipped from -hw to hw
        if abs(world_ray[0]) > 1e-6:
            t = (hw - position[0]) / world_ray[0]
            if t > 0:
                point = position + t * world_ray
                if abs(point[2]) <= hd and 0 <= point[1] <= self.room_height:
                    print(f"Left wall intersection at: {point}")

        # Test right wall (x = -hw)  # Flipped from hw to -hw
        if abs(world_ray[0]) > 1e-6:
            t = (-hw - position[0]) / world_ray[0]
            if t > 0:
                point = position + t * world_ray
                if abs(point[2]) <= hd and 0 <= point[1] <= self.room_height:
                    print(f"Right wall intersection at: {point}")
