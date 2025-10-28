# camera.py - カメラ関連の関数

import cv2
from config import DEFAULT_CAMERA_INDEX, DEFAULT_WIDTH, DEFAULT_HEIGHT, WINDOW_NAME

def initialize_camera(camera_index=DEFAULT_CAMERA_INDEX):
    """カメラを初期化し、解像度を設定する"""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)

    # 実際の解像度を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, width, height

def initialize_window(width, height):
    """ウィンドウを初期化する（サイズ変更不可）"""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    return False  # is_fullscreen

def get_frame(cap):
    """カメラからフレームを取得する"""
    ret, frame = cap.read()
    return ret, frame

def show_frame(frame):
    """フレームをウィンドウに表示する"""
    cv2.imshow(WINDOW_NAME, frame)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

def cleanup(cap):
    """リソースを解放する"""
    cap.release()
    cv2.destroyAllWindows()