# main.py - メインアプリケーション

import cv2
import argparse
from camera import initialize_camera, initialize_window, get_frame, show_frame, cleanup
from hand_detector import initialize_hand_detector, detect_hands, draw_hand_landmarks
from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="カメラデバイス番号")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # カメラの初期化
    cap, width, height = initialize_camera(args.device)

    # ウィンドウの初期化
    is_fullscreen = initialize_window(width, height)

    # 手の検出器の初期化
    detector = initialize_hand_detector()

    # FPS計算の初期化
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    try:
        while cap.isOpened():
            # FPSを取得
            fps = cv_fps_calc.get()

            # フレームを取得
            ret, frame = get_frame(cap)
            if not ret:
                break

            # 手の検出
            detection_result, image = detect_hands(detector, frame)

            # ランドマークを描画（FPS表示付き）
            image = draw_hand_landmarks(image, detection_result, fps)

            # 画面に表示
            show_frame(image)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # リソースの解放
        cleanup(cap)

if __name__ == "__main__":
    main()