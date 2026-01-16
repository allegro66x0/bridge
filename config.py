import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')

# Default Settings (Fallback)
defaults = {
    "SERIAL_PORT_GANTRY": "COM9",
    "SERIAL_PORT_MAGNET": "COM10", # Placeholder for Magnet Arduino
    "SERIAL_PORT_SORTER": "COM7",
    "SORTER_CAM_ID": 2,
    "SORTER_MODEL_PATH": "model.pth",
    "SORTER_ROI_PATH": "model_roi.json",
    "CAMERA_INDEX": 0,
    "SCREEN_WIDTH": 2880,
    "SCREEN_HEIGHT": 1920,
    "BOARD_SIZE_AI": 13,
    "SEARCH_DEPTH": 5,
    "FIXED_CORNER_POINTS": [[276, 234], [405, 229], [432, 376], [264, 379]]
}

# Load Config
settings = defaults.copy()
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            loaded = json.load(f)
            settings.update(loaded)
    except Exception as e:
        print(f"⚠️ Error loading config.json: {e}")

# Expose settings as module-level variables
SERIAL_PORT_GANTRY = settings["SERIAL_PORT_GANTRY"]
SERIAL_PORT_MAGNET = settings["SERIAL_PORT_MAGNET"]
SERIAL_PORT_SORTER = settings["SERIAL_PORT_SORTER"]

SORTER_CAM_ID = settings["SORTER_CAM_ID"]
SORTER_MODEL_PATH = settings["SORTER_MODEL_PATH"]
SORTER_ROI_PATH = settings["SORTER_ROI_PATH"]

CAMERA_INDEX = settings["CAMERA_INDEX"]
SCREEN_WIDTH = settings["SCREEN_WIDTH"]
SCREEN_HEIGHT = settings["SCREEN_HEIGHT"]

BOARD_SIZE_AI = settings["BOARD_SIZE_AI"]
SEARCH_DEPTH = settings["SEARCH_DEPTH"]

FIXED_CORNER_POINTS = [tuple(pt) for pt in settings["FIXED_CORNER_POINTS"]]

L6_SCRIPT_PATH = os.path.join(BASE_DIR, 'L6', 'webcam_gomoku_ai.py')
