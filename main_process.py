import os
import cv2
import argparse
from DanceTrack.yolo_pose import YOLOPose
from DanceTrack.dancer_tracker import DancerTracker
from DanceTrack.dance_room_tracker import DanceRoomTracker
from DanceTrack.normalize_poses import PoseNormalizer
from DanceTrack.temporal_smoothing import TemporalSmoothing
from DanceTrack.manual_review import ManualReview
from DanceTrack.foot_projector import FootProjector
from DanceTrack.debug_video import DebugVideo

def main(video_path:str, output_dir:str, room_dimension:str):


    # Get input video dimensions
    cap = cv2.VideoCapture(video_path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()



    # ROOM TRACKING
    parsed_room_dimension = parse_dimensions(room_dimension)
    danceRoomTracker = DanceRoomTracker(video_path, output_dir, parsed_room_dimension, frame_height, frame_width, frame_count)
    danceRoomTracker.run_video_loop()



def parse_dimensions(dimensions_str:str):
    try:
        # Split the input string by commas and convert each to a float
        w, d, h = map(float, dimensions_str.split(','))
        # Return as a tuple (x, y, z) where x = width, y = height, z = depth
        return w, h, d
    except ValueError:
        raise ValueError("Invalid input format. Please provide a comma-separated string of three float values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("--video_path", help="Path to the input video file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    parser.add_argument("--room_dimension", help="Width, Depth, Height")
    args = parser.parse_args()

    main(args.video_path, args.output_dir, args.room_dimension)

    #normalized_video_path = args.video_path
    #debug_video = DebugVideo(normalized_video_path, args.output_dir)
    #debug_video = DebugVideo(args.video_path, args.output_dir)
    #debug_video.generate_debug_video()
