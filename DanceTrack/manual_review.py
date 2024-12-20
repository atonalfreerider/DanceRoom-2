import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from collections import OrderedDict
import os
import time

import utils
from DanceTrack.pose_data_utils import PoseDataUtils


class ManualReview:
    def __init__(self, video_path:str, output_dir:str, frame_height:int, frame_width:int, frame_count:int):
        self.__detections_modified_file = os.path.join(output_dir, 'detections-modified.json')
        self.__cap = cv2.VideoCapture(video_path)
        self.__frame_count = frame_count
        self.current_frame = 0
        self.playing = False
        self.__window_name = "Manual Review"
        self.__button_height = 40
        self.__button_width = 150
        self.__button_color = (200, 200, 200)
        self.__button_text_color = (0, 0, 0)        
        self.__click_radius = 10
        self.__hovered_pose = None
        self.__dragging_keypoint = None
        self.frame_cache = OrderedDict()
        self.max_cache_size = 100
        self.__frame_height, self.__frame_width = frame_height, frame_width
        self.__ui_overlay = np.zeros((self.__frame_height, self.__frame_width, 3), dtype=np.uint8)
        self.__draw_ui_overlay()

        self.pose_utils = PoseDataUtils()

        # Load detections

        if Path(self.__detections_modified_file).exists():
            self.__detections = utils.load_json_integer_keys(self.__detections_modified_file)
        else:
            self.__detections = utils.load_json_integer_keys(os.path.join(output_dir, 'detections.json'))
            # If detections-modified.json doesn't exist, create it from detections.json
            utils.save_json(self.__detections, self.__detections_modified_file)

        # Load lead and follow
        self.__lead_file = os.path.join(output_dir, 'lead.json')
        self.__follow_file = os.path.join(output_dir, 'follow.json')
        self.__lead = utils.load_json_integer_keys(self.__lead_file) if Path(self.__lead_file).exists() else {}
        self.__follow = utils.load_json_integer_keys(self.__follow_file) if Path(self.__follow_file).exists() else {}

        # GUI
        root = tk.Tk()
        root.title("Save JSON")
        root.geometry("200x50")
        save_button = tk.Button(root, text="Save to JSON", command=self.__save_json_files)
        save_button.pack(pady=10)
        root.withdraw()  # Hide the window initially

        cv2.namedWindow(self.__window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.__window_name, self.__frame_width, self.__frame_height)
        cv2.setMouseCallback(self.__window_name, self.__mouse_callback)
        self.__create_trackbar()

    def __save_json_files(self):
        try:
            # Check if lead and follow data are not empty before saving
            if self.__lead:
                PoseDataUtils.save_poses(self.__lead, self.__frame_count, self.__lead_file)
                print(f"Saved lead tracks to {self.__lead_file}")
            else:
                print("Lead data is empty. Skipping save for lead.")

            if self.__follow:
                PoseDataUtils.save_poses(self.__follow, self.__frame_count, self.__follow_file)
                print(f"Saved follow tracks to {self.__follow_file}")
            else:
                print("Follow data is empty. Skipping save for follow.")

            # Save detections-modified.json
            if self.__detections:
                utils.save_json(self.__detections, self.__detections_modified_file)
                print(f"Saved modified detections to {self.__detections_modified_file}")
            else:
                print("Detections data is empty. Skipping save for detections.")

            print("Save completed successfully")
        except Exception as e:
            print(f"Error during save: {str(e)}")

    def __get_frame(self, frame_idx):
        if frame_idx not in self.frame_cache:
            self.__cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.__cap.read()
            if ret:
                if len(self.frame_cache) >= self.max_cache_size:
                    self.frame_cache.popitem(last=False)
                self.frame_cache[frame_idx] = frame
            else:
                return None
        return self.frame_cache[frame_idx]

    def __draw_frame(self):
        frame = self.__get_frame(self.current_frame)
        
        if frame is None:
            return

        frame = frame.copy()  # Create a copy to avoid modifying the original

        # Draw all poses from detections
        for detection in self.__detections.get(self.current_frame, []):
            pose_type = 'unknown'
            if self.current_frame in self.__lead and self.__lead[self.current_frame] and detection['id'] == self.__lead[self.current_frame]['id']:
                pose_type = 'lead'
                lead_bbox = detection['bbox']
                cv2.rectangle(frame, (int(lead_bbox[0]), int(lead_bbox[1])), (int(lead_bbox[2]), int(lead_bbox[3])), (0,0,255), 1)
            elif self.current_frame in self.__follow and self.__follow[self.current_frame] and detection['id'] == self.__follow[self.current_frame]['id']:
                pose_type = 'follow'
                follow_bbox = detection['bbox']
                cv2.rectangle(frame, (int(follow_bbox[0]), int(follow_bbox[1])), (int(follow_bbox[2]), int(follow_bbox[3])),
                              (255, 255, 255), 1)
            PoseDataUtils.draw_pose(frame, detection['keypoints'], detection['id'], pose_type)

        # Highlight hovered pose
        if self.__hovered_pose:
            PoseDataUtils.draw_pose(frame, self.__hovered_pose['keypoints'], self.__hovered_pose['id'], 'hovered')

        # Draw the "Save to JSON" button
        button_top = self.__frame_height - self.__button_height - 10
        button_left = self.__frame_width - self.__button_width - 10
        cv2.rectangle(
            frame,
            (button_left, button_top),
            (button_left + self.__button_width, button_top + self.__button_height),
            self.__button_color, -1)

        cv2.putText(frame, "Save to JSON", (button_left + 10, button_top + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.__button_text_color, 2)

        # Display current frame number
        cv2.putText(frame, f"Frame: {self.current_frame}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Update trackbar position
        cv2.setTrackbarPos('Frame', self.__window_name, self.current_frame)

        # Add the UI overlay to the frame
        frame = cv2.addWeighted(frame, 1, self.__ui_overlay, 1, 0)

        # Resize the frame to fit the screen if necessary
        if frame.shape[0] != self.__frame_height or frame.shape[1] != self.__frame_width:
            frame = cv2.resize(frame, (self.__frame_width, self.__frame_height))

        cv2.imshow(self.__window_name, frame)

    def __find_closest_keypoint(self, x, y):
        closest_distance = float('inf')
        closest_pose = None
        closest_keypoint_index = None

        for detection in self.__detections.get(self.current_frame, []):
            for i, keypoint in enumerate(detection['keypoints']):
                distance = np.sqrt((x - keypoint[0])**2 + (y - keypoint[1])**2)
                if distance < closest_distance and distance < self.__click_radius:
                    closest_distance = distance
                    closest_pose = detection
                    closest_keypoint_index = i

        return closest_pose, closest_keypoint_index

    def __mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            old_hovered_pose = self.__hovered_pose
            self.__hovered_pose, _ = self.__find_closest_keypoint(x, y)
            
            if self.__dragging_keypoint:
                pose, keypoint_index = self.__dragging_keypoint
                if pose is not None and keypoint_index is not None:
                    pose['keypoints'][keypoint_index][0] = x
                    pose['keypoints'][keypoint_index][1] = y
                    
                    # Update the detection in self.detections
                    frame_detections = self.__detections.get(self.current_frame, [])
                    for i, detection in enumerate(frame_detections):
                        if detection['id'] == pose['id']:
                            frame_detections[i] = pose
                            break
                    self.__detections[self.current_frame] = frame_detections
                
            # Redraw the frame if the hovered pose changed or if we're dragging
            if self.__hovered_pose != old_hovered_pose or self.__dragging_keypoint:
                self.__draw_frame()

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is on the "Save to JSON" button
            button_top = self.__frame_height - self.__button_height - 10
            button_left = self.__frame_width - self.__button_width - 10
            if button_left <= x <= button_left + self.__button_width and button_top <= y <= button_top + self.__button_height:
                self.__save_json_files()
                return

            self.__dragging_keypoint = self.__find_closest_keypoint(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.__dragging_keypoint = None
            self.__draw_frame()  # Redraw the frame when we stop dragging

    def __assign_role(self, role):
        if not self.__hovered_pose:
            return

        current_frame = int(self.current_frame)
        track_id = self.__hovered_pose['id']
        
        # Get the current role of the hovered pose (if any)
        current_role = None
        if current_frame in self.__lead and self.__lead[current_frame] and self.__lead[current_frame]['id'] == track_id:
            current_role = 'lead'
        elif current_frame in self.__follow and self.__follow[current_frame] and self.__follow[current_frame]['id'] == track_id:
            current_role = 'follow'

        # Get the other track's ID if there's a pose in the target role
        other_track_id = None
        if role == 'lead' and current_frame in self.__follow and self.__follow[current_frame]:
            other_track_id = self.__follow[current_frame]['id']
        elif role == 'follow' and current_frame in self.__lead and self.__lead[current_frame]:
            other_track_id = self.__lead[current_frame]['id']

        # Determine if we're swapping roles
        is_swapping = (
            (role == 'lead' and current_role == 'follow') or 
            (role == 'follow' and current_role == 'lead') or
            other_track_id is not None
        )

        # Iterate through subsequent frames
        for frame in range(current_frame, self.__frame_count):
            # Get all poses for this frame
            frame_poses = self.__detections.get(frame, [])
            
            # Find our track's pose in this frame
            track_pose = next((pose for pose in frame_poses if pose['id'] == track_id), None)
            if not track_pose:
                break  # Stop if our track is not in this frame

            # If we're swapping roles
            if is_swapping:
                # Find the other track's pose
                other_track_pose = next((pose for pose in frame_poses if pose['id'] == other_track_id), None)
                if not other_track_pose:
                    break  # Stop if the other track is not in this frame

                # Check if either pose has a different role assignment
                if frame > current_frame:  # Skip checking the first frame
                    if (frame in self.__lead and self.__lead[frame] and 
                        self.__lead[frame]['id'] not in [track_id, other_track_id]):
                        break
                    if (frame in self.__follow and self.__follow[frame] and 
                        self.__follow[frame]['id'] not in [track_id, other_track_id]):
                        break

                # Perform the swap
                if role == 'lead':
                    self.__lead[frame] = track_pose
                    self.__follow[frame] = other_track_pose
                else:  # role == 'follow'
                    self.__follow[frame] = track_pose
                    self.__lead[frame] = other_track_pose

            # If we're just assigning a new role
            else:
                # Check if this pose already has a different role assignment
                if frame > current_frame:  # Skip checking the first frame
                    existing_role = None
                    if frame in self.__lead and self.__lead[frame] and self.__lead[frame]['id'] == track_id:
                        existing_role = 'lead'
                    elif frame in self.__follow and self.__follow[frame] and self.__follow[frame]['id'] == track_id:
                        existing_role = 'follow'
                    
                    if existing_role and existing_role != role:
                        break  # Stop if we encounter a different role assignment

                # Assign the role
                if role == 'lead':
                    self.__lead[frame] = track_pose
                    # Remove from follow if it was there
                    if frame in self.__follow and self.__follow[frame] and self.__follow[frame]['id'] == track_id:
                        del self.__follow[frame]
                else:  # role == 'follow'
                    self.__follow[frame] = track_pose
                    # Remove from lead if it was there
                    if frame in self.__lead and self.__lead[frame] and self.__lead[frame]['id'] == track_id:
                        del self.__lead[frame]

        # Redraw the frame after all assignments
        self.__draw_frame()

    @staticmethod
    def __mirror_pose(pose):
        # Define the pairs of keypoints to be swapped
        swap_pairs = [
            (1, 2),   # Left Eye, Right Eye
            (3, 4),   # Left Ear, Right Ear
            (5, 6),   # Left Shoulder, Right Shoulder
            (7, 8),   # Left Elbow, Right Elbow
            (9, 10),  # Left Wrist, Right Wrist
            (11, 12), # Left Hip, Right Hip
            (13, 14), # Left Knee, Right Knee
            (15, 16)  # Left Ankle, Right Ankle
        ]

        for left, right in swap_pairs:
            # Swap the positions and confidence values
            pose['keypoints'][left], pose['keypoints'][right] = pose['keypoints'][right], pose['keypoints'][left]

        return pose

    def run(self):
        try:
            while True:
                self.__draw_frame()
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                elif key == 32:  # Spacebar
                    self.playing = not self.playing
                elif key == 83:  # Right arrow
                    self.current_frame = min(self.current_frame + 1, self.__frame_count - 1)
                elif key == 81:  # Left arrow
                    self.current_frame = max(self.current_frame - 1, 0)
                elif key == ord('1'):
                    self.__assign_role('lead')
                elif key == ord('2'):
                    self.__assign_role('follow')
                elif key == 13:  # Enter key
                    # Get the current value from the trackbar
                    new_frame = cv2.getTrackbarPos('Frame', self.__window_name)
                    self.current_frame = max(0, min(new_frame, self.__frame_count - 1))
                elif key == ord('r'):  # 'R' key for mirroring
                    if self.__hovered_pose:
                        # Mirror the hovered pose
                        mirrored_pose = self.__mirror_pose(self.__hovered_pose.copy())
                        
                        # Update the pose in self.detections
                        frame_detections = self.__detections.get(self.current_frame, [])
                        for i, detection in enumerate(frame_detections):
                            if detection['id'] == self.__hovered_pose['id']:
                                frame_detections[i] = mirrored_pose
                                break
                        self.__detections[self.current_frame] = frame_detections
                        
                        # Update the pose in lead or follow if it's assigned
                        if self.current_frame in self.__lead and self.__lead[self.current_frame] and self.__lead[self.current_frame]['id'] == self.__hovered_pose['id']:
                            self.__lead[self.current_frame] = mirrored_pose
                        elif self.current_frame in self.__follow and self.__follow[self.current_frame] and self.__follow[self.current_frame]['id'] == self.__hovered_pose['id']:
                            self.__follow[self.current_frame] = mirrored_pose
                        
                        # Update the hovered pose
                        self.__hovered_pose = mirrored_pose
                        
                        self.__draw_frame()
                elif key == ord('t'):  # 'T' key to add new T-pose
                    self.__add_t_pose()
                elif key == ord('0'):  # '0' key to unassign the hovered pose
                    self.__unassign_pose()
                elif key == 0x70:  # F1 key (0x70 is the scan code for F1)
                    self.__save_json_files()

                if self.playing:
                    self.current_frame = min(self.current_frame + 1, self.__frame_count - 1)
                    if self.current_frame == self.__frame_count - 1:
                        self.playing = False

                if cv2.getWindowProperty(self.__window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Add a small delay to reduce CPU usage
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Manual review interrupted. Cleaning up...")
        finally:
            self.__cap.release()
            cv2.destroyAllWindows()

    def __create_trackbar(self):
        cv2.createTrackbar('Frame', self.__window_name, 0, self.__frame_count - 1, self.__on_trackbar)

    def __on_trackbar(self, value):
        self.current_frame = value
        self.__draw_frame()

    def __draw_ui_overlay(self):
        # Clear the previous overlay
        self.__ui_overlay.fill(0)
        
        # Calculate button position based on frame dimensions
        button_top = self.__frame_height - self.__button_height - 10
        button_left = self.__frame_width - self.__button_width - 10
        
        # Draw the "Save to JSON" button on the UI overlay
        cv2.rectangle(self.__ui_overlay, (button_left, button_top), 
                      (button_left + self.__button_width, button_top + self.__button_height), 
                      self.__button_color, -1)
        cv2.putText(self.__ui_overlay, "Save to JSON", (button_left + 10, button_top + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.__button_text_color, 2)

    def __unassign_pose(self):
        if self.__hovered_pose:
            # Check if the hovered pose is assigned as lead
            if (self.current_frame in self.__lead and
                    self.__lead[self.current_frame] and
                    self.__lead[self.current_frame]['id'] == self.__hovered_pose['id']):
                del self.__lead[self.current_frame]

            # Check if the hovered pose is assigned as follow
            if (self.current_frame in self.__follow and
                    self.__follow[self.current_frame] and
                    self.__follow[self.current_frame]['id'] == self.__hovered_pose['id']):
                del self.__follow[self.current_frame]

            self.__draw_frame()

    def __create_t_pose(self):
        # Create a T-pose in the center of the frame, facing the camera
        center_x, center_y = self.__frame_width // 2, self.__frame_height // 2
        t_pose = {
            'id': -1,  # Use -1 as the track ID for manually added poses
            'bbox': [0, 0, 0, 0],  # Zeroed-out bounding box [x, y, width, height]
            'confidence': 0,  # Zero confidence score
            'keypoints': [
                [center_x, center_y - 100, 1],  # Nose
                [center_x - 15, center_y - 110, 1],  # Left Eye
                [center_x + 15, center_y - 110, 1],  # Right Eye
                [center_x - 25, center_y - 105, 1],  # Left Ear
                [center_x + 25, center_y - 105, 1],  # Right Ear
                [center_x - 80, center_y - 50, 1],  # Left Shoulder
                [center_x + 80, center_y - 50, 1],  # Right Shoulder
                [center_x - 150, center_y - 50, 1],  # Left Elbow
                [center_x + 150, center_y - 50, 1],  # Right Elbow
                [center_x - 220, center_y - 50, 1],  # Left Wrist
                [center_x + 220, center_y - 50, 1],  # Right Wrist
                [center_x - 30, center_y + 100, 1],  # Left Hip
                [center_x + 30, center_y + 100, 1],  # Right Hip
                [center_x - 30, center_y + 200, 1],  # Left Knee
                [center_x + 30, center_y + 200, 1],  # Right Knee
                [center_x - 30, center_y + 300, 1],  # Left Ankle
                [center_x + 30, center_y + 300, 1],  # Right Ankle
            ]
        }
        return t_pose

    def __add_t_pose(self):
        new_pose = self.__create_t_pose()
        
        # Add the new pose to the detections for the current frame
        frame_key = self.current_frame
        if frame_key not in self.__detections:
            self.__detections[frame_key] = []
        self.__detections[frame_key].append(new_pose)
        
        # Set the new pose as the hovered pose
        self.__hovered_pose = new_pose
        
        # Redraw the frame
        self.__draw_frame()
