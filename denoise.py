# denoise.py - ランドマークのデノイズ処理

import math

class LandmarkFilter:
    """ランドマークの座標を閾値ベースでフィルタリングするクラス"""

    def __init__(self, threshold=0.01):
        """
        初期化
        :param threshold: 移動距離の閾値（正規化座標での距離）
        """
        self.threshold = threshold
        self.previous_landmarks = {}  # ランドマークIDごとの前回位置

    def filter_landmarks(self, hand_landmarks):
        """
        ランドマークの座標をフィルタリングする
        :param hand_landmarks: 検出された手のランドマークリスト
        :return: フィルタリングされたランドマークリスト
        """
        filtered_landmarks = []

        for hand_idx, hand in enumerate(hand_landmarks):
            filtered_hand = []
            for landmark_idx, landmark in enumerate(hand):
                key = f"{hand_idx}_{landmark_idx}"

                # 前回のランドマークを取得
                prev_landmark = self.previous_landmarks.get(key)

                if prev_landmark is None:
                    # 初回は現在のランドマークを使用
                    filtered_hand.append(landmark)
                    self.previous_landmarks[key] = landmark
                else:
                    # 距離を計算
                    distance = math.sqrt(
                        (landmark.x - prev_landmark.x) ** 2 +
                        (landmark.y - prev_landmark.y) ** 2 +
                        (landmark.z - prev_landmark.z) ** 2
                    )

                    if distance > self.threshold:
                        # 閾値を超えた移動は反映
                        filtered_hand.append(landmark)
                        self.previous_landmarks[key] = landmark
                    else:
                        # 閾値以下の移動は無視（前回の位置を維持）
                        filtered_hand.append(prev_landmark)

            filtered_landmarks.append(filtered_hand)

        return filtered_landmarks

# グローバルなフィルターインスタンス
filter_instance = LandmarkFilter(threshold=0.005)
