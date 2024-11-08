import numpy as np
import json

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_json_integer_keys(json_path):
    with open(json_path, 'r') as f:
        detections = json.load(f)

    # Convert all frame keys to integers
    return {int(frame): data for frame, data in detections.items()}

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def save_numpy_json(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(numpy_to_python(data), f, indent=2)

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj