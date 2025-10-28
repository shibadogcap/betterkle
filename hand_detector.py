# hand_detector.py - 手の検出関連の関数

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from config import MAX_NUM_HANDS, MODEL_URL, MODEL_DIR, MODEL_NAME
from utils import download_file
import os
import absl.logging

# abslログの設定
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.ERROR)

def initialize_hand_detector():
    """MediaPipe HandLandmarkerを初期化する"""
    # モデルファイルのダウンロード
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        download_file(MODEL_URL, model_path)

    # HandLandmarkerの作成
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=MAX_NUM_HANDS,
        running_mode=RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    return detector

def detect_hands(detector, frame):
    """フレームから手のランドマークを検出する"""
    # MediaPipe Imageに変換
    rgb_frame = mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
    )

    # 検出
    detection_result = detector.detect(rgb_frame)

    return detection_result, frame

def draw_hand_landmarks(image, detection_result, fps=None):
    """検出された手のランドマークを描画する"""
    image_width, image_height = image.shape[1], image.shape[0]

    for hand_landmarks in detection_result.hand_landmarks:
        # ランドマークを描画
        for landmark in hand_landmarks:
            if landmark.visibility < 0 or landmark.presence < 0:
                continue
            landmark_x = int(landmark.x * image_width)
            landmark_y = int(landmark.y * image_height)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

        # 接続線を描画
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 親指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 人差し指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 薬指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        ]
        for connection in connections:
            start_idx, end_idx = connection
            start_point = (int(hand_landmarks[start_idx].x * image_width), int(hand_landmarks[start_idx].y * image_height))
            end_point = (int(hand_landmarks[end_idx].x * image_width), int(hand_landmarks[end_idx].y * image_height))
            cv2.line(image, start_point, end_point, (220, 220, 220), 2)

    # FPSを表示
    if fps is not None:
        cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return image