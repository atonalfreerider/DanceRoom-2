import os
import cv2
from tqdm import tqdm
import utils
from pose_data_utils import PoseDataUtils


class DebugVideo:
    def __init__(self, input_path, output_dir):
        self.__input_path = input_path
        self.__output_dir = output_dir
        self.__lead_file = os.path.join(output_dir, 'lead-normalized.json')
        self.__follow_file = os.path.join(output_dir, 'follow-normalized.json')
        self.__lead = utils.load_json(self.__lead_file)
        self.__follow = utils.load_json(self.__follow_file)


    def generate_debug_video(self):
        debug_video_path = os.path.join(self.__output_dir, 'debug_video.mp4')
        cap = cv2.VideoCapture(self.__input_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(debug_video_path, fourcc, fps, (frame_width, frame_height))

        # Load tracked sequences
        lead_track = utils.load_json(self.__lead_file)
        follow_track = utils.load_json(self.__follow_file)

        # Use tqdm to track progress
        with tqdm(total=total_frames, desc="Generating debug video") as pbar:
            for frame_count in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw lead and follow
                lead_pose = lead_track.get(str(frame_count))
                if lead_pose:
                    lead_keypoints = lead_pose['keypoints']
                    PoseDataUtils.draw_pose(frame, lead_keypoints, lead_pose[id], 'lead')


                follow_pose = follow_track.get(str(frame_count))
                if follow_pose:
                    follow_keypoints = follow_pose['keypoints']
                    PoseDataUtils.draw_pose(frame, follow_keypoints, follow_pose['id'],  'follow')

                out.write(frame)

                # Update progress bar
                pbar.update(1)

        cap.release()
        out.release()
