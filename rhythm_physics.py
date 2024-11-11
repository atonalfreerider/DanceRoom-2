import os
import argparse
from pathlib import Path

import utils

def main(output_dir:str):
    ground_ankle_path = os.path.join(output_dir, 'all_floor_ankles.json')
    ground_ankles = utils.load_json(ground_ankle_path)

    lead_keypoints_path = os.path.join(output_dir, 'lead_smoothed_keypoints_3d.json')
    lead_keypoints = utils.load_json(lead_keypoints_path)

    follow_keypoints_path = os.path.join(output_dir, 'follow_smoothed_keypoints_3d.json')
    follow_keypoints = utils.load_json(follow_keypoints_path)

    dir_path = Path(output_dir)
    dir_name = dir_path.stem

    rhythm_path = os.path.join(output_dir, f'{dir_name}_zouk-time-analysis.json')
    rhythm = utils.load_json(rhythm_path)

    x = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align 3D poses, and compute rhythm physics, patterns.")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    main(args.output_dir)
