# config.py - 設定ファイル

# ウィンドウ設定
WINDOW_NAME = 'BetterKLE'

# カメラ設定
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540

# MediaPipe設定
MAX_NUM_HANDS = 2
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# モデル設定
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_DIR = 'model'
MODEL_NAME = 'hand_landmarker_float16_1.task'