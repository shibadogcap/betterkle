# video.py - 動画ファイルからの映像入力

import cv2
from config import DEFAULT_WIDTH, DEFAULT_HEIGHT, WINDOW_NAME

def initialize_video(video_path):
    """動画ファイルを初期化し、解像度を設定する"""
    cap = cv2.VideoCapture(video_path)

    # 実際の解像度を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, width, height

def initialize_window(width, height):
    """ウィンドウを初期化する（サイズ変更不可）"""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    return False  # is_fullscreen

def get_frame(cap):
    """動画からフレームを取得する"""
    ret, frame = cap.read()
    return ret, frame

def show_frame(frame):
    """フレームをウィンドウに表示する"""
    cv2.imshow(WINDOW_NAME, frame)

def cleanup(cap):
    """リソースを解放する"""
    cap.release()
    cv2.destroyAllWindows()
