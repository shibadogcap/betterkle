# main.py - メインアプリケーション

import cv2
import argparse
import time
import matplotlib.pyplot as plt
from camera import initialize_camera, initialize_window, get_frame, show_frame, cleanup
from video import initialize_video
from hand_detector import initialize_hand_detector, detect_hands, draw_hand_landmarks
from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="カメラデバイス番号")
    parser.add_argument("--video", type=str, help="動画ファイルのパス")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # カメラまたは動画の初期化
    if args.video:
        cap, width, height, video_fps = initialize_video(args.video)
        frame_time = 1 / video_fps if video_fps > 0 else 0
        prev_time = time.time()
    else:
        cap, width, height = initialize_camera(args.device)
        frame_time = 0  # カメラの場合は制御しない
        prev_time = time.time()

    # ウィンドウの初期化
    is_fullscreen = initialize_window(width, height)

    # 手の検出器の初期化
    detector = initialize_hand_detector()

    # FPS計算の初期化
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    # ランドマークの軌跡を保存するリスト
    landmark_trajectories = []

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

            # ランドマークの軌跡を保存
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    trajectory = [(lm.x, lm.y) for lm in hand_landmarks]
                    landmark_trajectories.append(trajectory)

            # 画面に表示
            show_frame(image)

            # キー入力の処理
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # キーが押された場合
                print(f"Pressed key: {chr(key) if key < 128 else 'unknown'}")
            if key == 27:  # ESCキーで終了
                break

            # ビデオの場合、フレームレートを制御
            if args.video and frame_time > 0:
                curr_time = time.time()
                elapsed = curr_time - prev_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                prev_time = curr_time
    finally:
        # リソースの解放
        cleanup(cap)

        # ランドマークの軌跡を描画
        if landmark_trajectories:
            plt.figure(figsize=(10, 10))
            for trajectory in landmark_trajectories:
                x_coords, y_coords = zip(*trajectory)
                plt.plot(x_coords, y_coords, marker='o')
            plt.title("Landmark Trajectories")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.gca().invert_yaxis()
            plt.show()

if __name__ == "__main__":
    main()