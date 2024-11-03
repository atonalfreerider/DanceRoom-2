import os
import argparse
import sam2
import yolo_pose
import dance_room_tracker

def main(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # possibly use segments. slow process at 1920x1080 x 3min: ~5hr
    # saM2 = sam2.Sam2(video_path, output_dir)
    # saM2.run()

    # use YOLOv11-pose to detect poses of figures and track them with IoU. Output detections.json for later use
    #yoloPose = yolo_pose.YOLOPose(video_path, output_dir)
    #yoloPose.detect_poses()

    # Initialize room tracking and dancer selection
    danceRoomTracker = dance_room_tracker.DanceRoomTracker(video_path, output_dir)
    danceRoomTracker.run_video_loop()  # New method for the updated workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for person segmentation and room orientation.")
    parser.add_argument("--video_path", help="Path to the input video file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.video_path, args.output_dir)
