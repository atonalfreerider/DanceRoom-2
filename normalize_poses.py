import numpy as np
import cv2
from tqdm import tqdm
import os
import utils


class PoseNormalizer:
    def __init__(self, video_path:str, output_dir:str):
        self.__video_path = video_path
        self.__output_dir = output_dir
        self.__camera_data = None
        self.__lead_data = None
        self.__follow_data = None
        self.__widest_focal = None
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        self.__frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.image_center = (self.__frame_width // 2, self.__frame_height // 2)

    @staticmethod
    def __find_widest_focal_length(camera_data):
        # Extract all non-negative focal lengths
        focal_lengths = [frame_data['focal_length'] for frame_data in camera_data.values() if
                         frame_data['focal_length'] >= 0]

        # Return the minimum (widest) focal length, or None if no non-negative focal length is found
        return min(focal_lengths) if focal_lengths else None

    def __normalize_keypoints(self, keypoints, current_focal, target_focal):
        """
        Normalize keypoints based on focal length change, scaling from image center
        """
        normalized_keypoints = []
        # Invert scale factor to match video normalization
        scale_factor = target_focal / current_focal
        
        for kp in keypoints:
            if kp[0] == 0 and kp[1] == 0:  # Skip invalid keypoints
                normalized_keypoints.append(kp)
                continue
                
            # Scale point relative to center
            x = kp[0]
            y = kp[1]
            confidence = kp[2]
            
            # Transform point relative to center
            x_centered = x - self.image_center[0]
            y_centered = y - self.image_center[1]
            
            # Scale
            x_scaled = x_centered * scale_factor
            y_scaled = y_centered * scale_factor
            
            # Transform back
            x_final = x_scaled + self.image_center[0]
            y_final = y_scaled + self.image_center[1]
            
            normalized_keypoints.append([x_final, y_final, confidence])
        
        return normalized_keypoints

    def __normalize_bbox(self, bbox, current_focal, target_focal):
        """
        Normalize bounding box coordinates based on focal length change
        """
        # Invert scale factor to match video normalization
        scale_factor = target_focal / current_focal
        
        x1, y1, x2, y2 = bbox
        
        # Transform and scale relative to center
        x1_centered = (x1 - self.image_center[0]) * scale_factor + self.image_center[0]
        y1_centered = (y1 - self.image_center[1]) * scale_factor + self.image_center[1]
        x2_centered = (x2 - self.image_center[0]) * scale_factor + self.image_center[0]
        y2_centered = (y2 - self.image_center[1]) * scale_factor + self.image_center[1]
        
        return [x1_centered, y1_centered, x2_centered, y2_centered]

    def __normalize_pose_data(self, pose_data, camera_data, target_focal):
        normalized_data = {}
        
        for frame_num in pose_data:
            if frame_num not in camera_data:
                continue
                
            current_focal = camera_data[frame_num]['focal_length']
            frame_data = pose_data[frame_num].copy()
            
            # Normalize keypoints
            frame_data['keypoints'] = self.__normalize_keypoints(
                frame_data['keypoints'],
                current_focal,
                target_focal
            )
            
            # Normalize bbox
            frame_data['bbox'] = self.__normalize_bbox(
                frame_data['bbox'],
                current_focal,
                target_focal
            )
            
            normalized_data[frame_num] = frame_data
        
        return normalized_data

    def __normalize_frame(self, frame, current_focal):
        """Scale frame based on focal length difference"""
        if current_focal <= 0 or self.__widest_focal <= 0:
            return frame  # Return original frame if we have invalid focal lengths
        
        # Invert the scale factor to cancel out zoom
        # When current_focal is larger than widest_focal (zoomed in), we want to scale down
        # When current_focal is smaller than widest_focal (zoomed out), we want to scale up
        scale_factor = self.__widest_focal / current_focal  # Inverted from previous version
        
        # Calculate new dimensions
        new_width = max(1, int(self.__frame_width * scale_factor))
        new_height = max(1, int(self.__frame_height * scale_factor))
        
        # Create black background
        normalized = np.zeros((self.__frame_height, self.__frame_width, 3), dtype=np.uint8)
        
        try:
            if scale_factor <= 1.0:
                # If scaling down, resize frame and center it
                resized = cv2.resize(frame, (new_width, new_height))
                y_offset = (self.__frame_height - new_height) // 2
                x_offset = (self.__frame_width - new_width) // 2
                normalized[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            else:
                # If scaling up, resize first then crop to center
                resized = cv2.resize(frame, (new_width, new_height))
                
                # Calculate crop coordinates to get the center portion
                y_start = (new_height - self.__frame_height) // 2
                x_start = (new_width - self.__frame_width) // 2
                y_end = y_start + self.__frame_height
                x_end = x_start + self.__frame_width
                
                # Ensure valid crop coordinates
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                y_end = min(new_height, y_end)
                x_end = min(new_width, x_end)
                
                # Crop to original size
                normalized = resized[y_start:y_end, x_start:x_end]
                
                # If the cropped region is smaller than the target size, pad with black
                if normalized.shape[0] < self.__frame_height or normalized.shape[1] < self.__frame_width:
                    temp = np.zeros((self.__frame_height, self.__frame_width, 3), dtype=np.uint8)
                    y_pad = (self.__frame_height - normalized.shape[0]) // 2
                    x_pad = (self.__frame_width - normalized.shape[1]) // 2
                    temp[y_pad:y_pad+normalized.shape[0], x_pad:x_pad+normalized.shape[1]] = normalized
                    normalized = temp
                
            return normalized
        except Exception as e:
            print(f"Error processing frame with scale_factor {scale_factor}: {str(e)}")
            return frame  # Return original frame if any error occurs

    def __process_video(self, output_path):
        cap = cv2.VideoCapture(self.__video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (self.__frame_width, self.__frame_height))

        with tqdm(total=total_frames, desc="Normalizing video") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get focal length for current frame
                current_focal = self.__camera_data.get(str(frame_idx), {}).get('focal_length')
                if current_focal is None:
                    break

                # Normalize frame
                normalized_frame = self.__normalize_frame(frame, current_focal)
                out.write(normalized_frame)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

    def __normalize_vo_data(self):
        """Normalize visual odometry data and combine with initial pose"""
        # Load initial camera pose
        initial_pose = utils.load_json(os.path.join(self.__output_dir, 'initial_camera_pose.json'))
        initial_position = initial_pose['position']
        
        normalized_vo = []
        
        # Iterate through all frames in camera data
        for frame_idx in sorted(self.__camera_data.keys(), key=int):
            frame_data = self.__camera_data[frame_idx]
            
            # Get rotation quaternion from frame data
            rotation = frame_data['rotation']
            
            # Create entry with initial position and frame rotation
            vo_entry = [
                round(float(initial_position[0]), 10),  # x
                round(float(initial_position[1]), 10),  # y
                round(float(initial_position[2]), 10),  # z
                round(float(rotation[0]), 10),  # q_x
                round(float(rotation[1]), 10),  # q_y
                round(float(rotation[2]), 10),  # q_z
                round(float(rotation[3]), 10)   # q_w
            ]
            
            normalized_vo.append(vo_entry)
        
        # Save normalized visual odometry data
        vo_normalized_path = os.path.join(self.__output_dir, 'vo_normalized.json')
        utils.save_json(normalized_vo, vo_normalized_path)
        print(f"4. Normalized visual odometry data: {vo_normalized_path}")

    def run(self):
        os.makedirs(self.__output_dir, exist_ok=True)

        # Load data
        self.__camera_data = utils.load_json(os.path.join(self.__output_dir, 'camera_tracking.json'))
        self.__lead_data = utils.load_json(os.path.join(self.__output_dir, 'lead.json'))
        self.__follow_data = utils.load_json(os.path.join(self.__output_dir, 'follow.json'))
        
        # Find widest focal length
        self.__widest_focal = self.__find_widest_focal_length(self.__camera_data)
        print(f"Widest focal length found: {self.__widest_focal}")
        
        # Normalize pose data
        lead_normalized = self.__normalize_pose_data(self.__lead_data, self.__camera_data, self.__widest_focal)
        follow_normalized = self.__normalize_pose_data(self.__follow_data, self.__camera_data, self.__widest_focal)

        self.__normalize_vo_data()
        
        # Save normalized pose data
        lead_normalized_path = os.path.join(self.__output_dir, 'lead-normalized.json')
        follow_normalized_path = os.path.join(self.__output_dir, 'follow-normalized.json')
        utils.save_json(lead_normalized, lead_normalized_path)
        utils.save_json(follow_normalized, follow_normalized_path)
        
        # Process video
        normalized_video_path = os.path.join(self.__output_dir, 'normalized_video.mp4')
        self.__process_video(normalized_video_path)
