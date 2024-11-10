import os
import argparse
from yolo_pose import YOLOPose
from dancer_tracker import DancerTracker
from dance_room_tracker import DanceRoomTracker
from normalize_poses import PoseNormalizer
from temporal_smoothing import TemporalSmoothing
from manual_review import ManualReview
from debug_video import DebugVideo

def main(video_path:str, output_dir:str, room_dimension:str):
    os.makedirs(output_dir, exist_ok=True)

    # DANCER TRACKING

    # use YOLOv11-pose to detect poses of figures and track them with IoU. Output detections.json for later use
    yoloPose = YOLOPose(video_path, output_dir)
    yoloPose.detect_poses()

    # use DeepFace (VGG-Face) to track dancers by face id
    dancerTracker = DancerTracker(video_path, output_dir)
    dancerTracker.process_video()

    manualReview = ManualReview(video_path, output_dir)
    manualReview.run()

    # ROOM TRACKING
    parsed_room_dimension = parse_dimensions(room_dimension)
    danceRoomTracker = DanceRoomTracker(video_path, output_dir, parsed_room_dimension)
    danceRoomTracker.run_video_loop()

    # NORMALIZE AND SMOOTH

    normalizer = PoseNormalizer(video_path, output_dir)
    normalizer.run()

    smoother = TemporalSmoothing(output_dir)
    smoother.run()

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
