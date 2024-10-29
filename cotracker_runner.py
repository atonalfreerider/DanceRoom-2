# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import cv2

from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

class CoTracker:
    def __init__(self):
        self.model = CoTrackerPredictor(
            checkpoint="checkpoints/scaled_offline.pth",
            window_len=60,
        )
        self.model = self.model.to(DEFAULT_DEVICE)


    def track(self, video_path, points, start_frame=0, num_frames=50):
        """Track points through video segment"""
        # Load video segment
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to RGB and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return None, None
            
        # Convert to tensor
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        video = video.unsqueeze(0).to(DEFAULT_DEVICE)

        # Convert points to queries format (t, x, y)
        queries = torch.zeros((1, len(points), 3), device=DEFAULT_DEVICE)
        for i, (x, y) in enumerate(points):
            queries[0, i] = torch.tensor([0, x, y])

        # Run tracking
        pred_tracks, pred_visibility = self.model(
            video,
            queries=queries,
            backward_tracking=True
        )

        return pred_tracks[0].cpu().numpy(), pred_visibility[0].cpu().numpy()

