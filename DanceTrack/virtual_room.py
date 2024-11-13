import numpy as np
import cv2


class VirtualRoom:
    def __init__(self, room_dimensions):
        # Room dimensions in meters
        self.__room_width = room_dimensions[0]
        self.__room_depth = room_dimensions[2]
        self.__room_height = room_dimensions[1]
        self.__grid_size = 1.0  # 1m^2 grid tiles
        self.__frame_width, self.__frame_height = None, None

    def set_frame(self, frame_height, frame_width):
        self.__frame_width = frame_width
        self.__frame_height = frame_height

    def project_point_to_planes(self, world_ray, camera_position):
        # Test intersections
        hw, hd = self.__room_width / 2, self.__room_depth / 2
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
                if abs(point[0]) <= hw and 0 <= point[1] <= self.__room_height:
                    intersections.append(('back_wall', point, t))

        # Test left wall (x = -hw)
        if abs(world_ray[0]) > 1e-6:
            t = (-hw - camera_position[0]) / world_ray[0]
            if t > 0:
                point = camera_position + t * world_ray
                if abs(point[2]) <= hd and 0 <= point[1] <= self.__room_height:
                    intersections.append(('left_wall', point, t))

        # Test right wall (x = hw)
        if abs(world_ray[0]) > 1e-6:
            t = (hw - camera_position[0]) / world_ray[0]
            if t > 0:
                point = camera_position + t * world_ray
                if abs(point[2]) <= hd and 0 <= point[1] <= self.__room_height:
                    intersections.append(('right_wall', point, t))

        # Return closest intersection point and its type
        if intersections:
            intersections.sort(key=lambda x: x[2])  # Sort by distance
            return intersections[0]  # Return the point

        return None

    def project_points(self, vertices, rotation, position, focal_length):
        fx = fy = min(self.__frame_height, self.__frame_width)
        cx, cy = self.__frame_width / 2, self.__frame_height / 2

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

        # Generate room vertices with origin at floor center
        hw, hd = self.__room_width / 2, self.__room_depth / 2
        h = self.__room_height  # Full height, since y=0 is now at floor level
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

        def clip_line(p1, p2, i1, i2):
            """Clip line to frame boundaries and check visibility"""
            # If either point is behind camera, don't draw
            if behind_camera[i1] or behind_camera[i2]:
                return False

            # If both points are outside frame in same direction, don't draw
            if p1 is not None and p2 is not None:
                if ((p1[0] < 0 and p2[0] < 0) or
                        (p1[0] >= self.__frame_width and p2[0] >= self.__frame_width) or
                        (p1[1] < 0 and p2[1] < 0) or
                        (p1[1] >= self.__frame_height and p2[1] >= self.__frame_height)):
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
        x_count = int(self.__room_width / self.__grid_size) - 1
        z_count = int(self.__room_depth / self.__grid_size) - 1

        def is_point_in_frame(point):
            if point is None:
                return False
            return (0 <= point[0] < self.__frame_width and
                    0 <= point[1] < self.__frame_height)

        # Create grid starting from back to front, left to right
        for i in range(z_count):
            z = -hd + self.__grid_size + i * self.__grid_size  # Start from back wall
            for j in range(x_count):
                x = hw - self.__grid_size - j * self.__grid_size  # Start from right wall (flipped x)

                # Define cross points in world coordinates (y=0 for floor)
                cross_center = np.array([x, 0, z])
                cross_points = [
                    cross_center,
                    cross_center + np.array([-cross_size, 0, 0]),  # Right (flipped)
                    cross_center - np.array([-cross_size, 0, 0]),  # Left (flipped)
                    cross_center + np.array([0, 0, cross_size]),  # Forward
                    cross_center - np.array([0, 0, cross_size])  # Back
                ]

                # Project all points of the cross
                cross_pixel_points, cross_behind = self.project_points(
                    cross_points,
                    rotation,
                    position,
                    focal_length)

                center = cross_pixel_points[0]
                all_visible = not any(cross_behind) and is_point_in_frame(center)

                # Draw cross if all points are visible
                if all_visible:
                    right = cross_pixel_points[1]
                    left = cross_pixel_points[2]
                    forward = cross_pixel_points[3]
                    back = cross_pixel_points[4]

                    cv2.line(frame, center, right, (0, 0, 255), 1)  # Red cross
                    cv2.line(frame, center, left, (0, 0, 255), 1)  # Red cross
                    cv2.line(frame, center, forward, (0, 0, 255), 1)  # Red cross
                    cv2.line(frame, center, back, (0, 0, 255), 1)  # Red cross

        origin_points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        projected_origin, origin_behind = self.project_points(
            origin_points,
            rotation,
            position,
            focal_length)

        origin_pt = projected_origin[0]
        origin_visible = not any(origin_behind) and is_point_in_frame(origin_pt)
        # Draw cross if all points are visible
        if origin_visible:
            x_origin = projected_origin[1]
            y_origin = projected_origin[2]
            z_origin = projected_origin[3]

            cv2.line(frame, origin_pt, x_origin, (0, 0, 255), 3)
            cv2.line(frame, origin_pt, y_origin, (0, 255, 0), 3)
            cv2.line(frame, origin_pt, z_origin, (255, 0, 0), 3)

        return frame
