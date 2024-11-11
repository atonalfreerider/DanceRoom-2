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
        self.__model = CoTrackerPredictor(
            checkpoint="checkpoints/scaled_offline.pth",
            window_len=60,
        )
        self.__model = self.__model.to(DEFAULT_DEVICE)

    def track(self, video_path, points, start_frame=0, num_frames=50):
        """Track points through video segment with automatic batch size adjustment"""
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

        # Try tracking with different batch sizes
        batch_sizes = [50, 25, 12]  # Progressively smaller batch sizes
        
        for batch_size in batch_sizes:
            try:
                if batch_size < len(frames):
                    # Process in batches
                    all_tracks = []
                    all_visibilities = []
                    
                    for i in range(0, len(frames), batch_size):
                        end_idx = min(i + batch_size, len(frames))
                        batch_video = video[:, i:end_idx]
                        
                        # Adjust queries for batch start time
                        batch_queries = queries.clone()
                        batch_queries[..., 0] = 0  # Reset time to 0 for each batch
                        
                        pred_tracks, pred_visibility = self.__model(
                            batch_video,
                            queries=batch_queries,
                            backward_tracking=True
                        )
                        
                        all_tracks.append(pred_tracks[0].cpu().numpy())
                        all_visibilities.append(pred_visibility[0].cpu().numpy())
                    
                    # Concatenate results
                    final_tracks = np.concatenate(all_tracks, axis=0)
                    final_visibilities = np.concatenate(all_visibilities, axis=0)
                    
                    return final_tracks, final_visibilities
                else:
                    # Process all frames at once
                    pred_tracks, pred_visibility = self.__model(
                        video,
                        queries=queries,
                        backward_tracking=True
                    )
                    return pred_tracks[0].cpu().numpy(), pred_visibility[0].cpu().numpy()
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error with batch size {batch_size}, trying smaller batch...")
                    torch.cuda.empty_cache()  # Clear CUDA memory
                    continue
                else:
                    raise e
        
        print("Failed to track with any batch size")
        return None, None

