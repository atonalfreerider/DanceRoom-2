import os
import argparse
import yolo_pose
from dancer_tracker import DancerTracker
import dance_room_tracker
from normalize_poses import PoseNormalizer
from temporal_smoothing import TemporalSmoothing
from debug_video import DebugVideo

def main(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # DANCER TRACKING

    # use YOLOv11-pose to detect poses of figures and track them with IoU. Output detections.json for later use
    yoloPose = yolo_pose.YOLOPose(video_path, output_dir)
    yoloPose.detect_poses()

    # use DeepFace (VGG-Face) to track dancers by face id
    dancer_tracker = DancerTracker(video_path, output_dir)
    dancer_tracker.process_video()

    #TODO user input to correct dancer tracks

    # ROOM TRACKING

    danceRoomTracker = dance_room_tracker.DanceRoomTracker(video_path, output_dir)
    danceRoomTracker.run_video_loop()

    # NORMALIZE AND SMOOTH

    normalizer = PoseNormalizer(video_path, output_dir)
    normalizer.run()

    smoother = TemporalSmoothing(output_dir)
    smoother.run()

    #debug_video = DebugVideo(normalized_video_path, output_dir)
    #debug_video = DebugVideo(video_path, output_dir)
    #debug_video.generate_debug_video()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("--video_path", help="Path to the input video file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.video_path, args.output_dir)
