import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import tqdm

import utils

# detects all poses inside a video (or loads from cache)
class YOLOPose:
    def __init__(self, video_path:str, output_dir:str):
        self.__video_path = video_path
        self.__detections_file = os.path.join(output_dir, 'detections.json')
        self.__detections = utils.load_json(self.__detections_file)

    def detect_poses(self):
        if self.__detections:
            return

        model = YOLO('yolo11x-pose.pt', task='pose')
        input_path = Path(self.__video_path)

        if input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            self.__process_video(model, self.__video_path)
        elif input_path.is_dir():
            self.__process_image_directory(model, input_path)
        else:
            raise ValueError("Input must be a video file or a directory containing PNG images.")

        utils.save_json(self.__detections, self.__detections_file)
        print(f"Saved pose detections for {len(self.__detections)} frames/images.")

    def __process_video(self, model, video_path:str):
        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pb = tqdm.tqdm(total=total_frames, desc="Processing poses from video frames with YOLO")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.__process_frame(model, frame, frame_count)
            frame_count += 1
            pb.update(1)

        cap.release()
        pb.close()

    def __process_image_directory(self, model, dir_path:Path):
        image_files = sorted([f for f in dir_path.glob('*.png')])
        total_images = len(image_files)

        pbar = tqdm.tqdm(total=total_images, desc="Processing images")
        for i, image_file in enumerate(image_files):
            frame = cv2.imread(str(image_file))
            self.__process_frame(model, frame, i)
            pbar.update(1)

        pbar.close()

    def __process_frame(self, model, frame, frame_index:int):
        results = model.track(frame, stream=True, persist=True, verbose=False)

        frame_detections = []

        for r in results:
            if r.boxes.id is None:
                ids = []
            else:
                ids = r.boxes.id.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            keypoints = r.keypoints.data.cpu().numpy()

            for track_id, box, conf, kps in zip(ids, boxes, confs, keypoints):
                detection = {
                    "id": int(track_id),
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "keypoints": kps.tolist()
                }
                frame_detections.append(detection)

        self.__detections[str(frame_index)] = frame_detections
