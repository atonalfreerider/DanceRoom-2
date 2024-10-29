# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

class CoTracker:
    def track(self, video_path, selected_points):

        model = CoTrackerPredictor(
            checkpoint="checkpoints/scaled_offline.pth",
            window_len=60,
        )

        model = model.to(DEFAULT_DEVICE)
        #todo video from openCV
        video = None

        first_frame = video[0, 0].permute(1, 2, 0).cpu().numpy()
        first_frame = (first_frame * 255).astype(np.uint8)

        if selected_points is None or len(selected_points) == 0:
            print("No points selected. Exiting.")
            exit()

        # Convert points to queries format (t, x, y)
        queries = torch.zeros((1, len(selected_points), 3), device=DEFAULT_DEVICE)
        for i, (x, y) in enumerate(selected_points):
            queries[0, i] = torch.tensor([0, x, y])

        # Run tracking with selected points
        chunk_size = 50  # Process 50 frames at a time to avoid out of memory
        all_tracks = []
        all_visibility = []

        for i in range(0, video.shape[1], chunk_size):
            video_chunk = video[:, i:i + chunk_size]

            if i == 0:
                # For the first chunk, use original queries
                chunk_queries = queries
            else:
                # For subsequent chunks, use the last known positions as starting points
                last_tracks = all_tracks[-1]
                last_positions = last_tracks[:, -1]  # Get positions from last frame of previous chunk
                chunk_queries = torch.zeros_like(queries)
                chunk_queries[:, :, 0] = 0  # Set time index to 0 for new chunk
                chunk_queries[:, :, 1:] = last_positions  # Use last known positions

            pred_tracks, pred_visibility = model(
                video_chunk,
                backward_tracking=True,
                segm_mask=None,
                queries=chunk_queries
            )

            # Store results
            all_tracks.append(pred_tracks)
            all_visibility.append(pred_visibility)

        # Concatenate results along time dimension
        pred_tracks = torch.cat(all_tracks, dim=1)
        pred_visibility = torch.cat(all_visibility, dim=1)

        print("computed")

        pred_tracks_numpy = pred_tracks[0].long().detach().cpu().numpy()
        pred_visibility_numpy = pred_visibility.cpu().numpy()

        return pred_tracks_numpy, pred_visibility_numpy

