import os
import json
from pathlib import Path


def load_json_config(filename='config/camera_config.json'):
    with open(filename) as f:
        data = json.load(f)
        return data


def overwrite_json_config(filename='config/camera_config.json', key="", value=""):
    with open(filename, 'r+') as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)  # rewind
        json.dump(data, f)
        f.truncate()
        print('Value correctly saved to JSON file')
