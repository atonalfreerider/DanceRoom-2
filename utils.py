import numpy as np
import json

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def save_numpy_json(data, filepath):
    with open(filepath, 'w') as f:
        # Convert NumPy array to list before saving to JSON
        if isinstance(data, np.ndarray):
            data = data.tolist()
        json.dump(data, f, indent=2)