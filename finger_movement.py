# mediapipeの出力から各指の動きを取得するために、手首の動きを補正
#         8   12  16  20
#         7   11  15  19
#         6   10  14  18
#         5   9   13  17
# 4 3 2 1     0
# (0: 手首, 1-4: 親指, 5-8: 人差し指, 9-12: 中指, 13-16: 薬指, 17-20: 小指)
# x, y, zについて指の付け根のランドマーク全体の移動量の平均から補正→x,yについては回転も考慮
import numpy as np
def get_general_movement(hand_landmarks):
    if not hand_landmarks:
        return None  # 手が検出されなかった場合

    movements = []

    for hand in hand_landmarks:
        # 手首(0)と各指の付け根(1,5,9,13,17)のランドマークを取得
        base_landmarks = [hand[0], hand[1], hand[5], hand[9], hand[13], hand[17]]

        # x, y, zの平均移動量を計算
        avg_x = np.mean([lm.x for lm in base_landmarks])
        avg_y = np.mean([lm.y for lm in base_landmarks])
        avg_z = np.mean([lm.z for lm in base_landmarks])

        movements.append((avg_x, avg_y, avg_z))

    return movements

def get_ronate_movement(hand_landmarks):
    if not hand_landmarks:
        return None  # 手が検出されなかった場合

    movements = []

    for hand in hand_landmarks:
        # 手首(0)と人差し指の付け根(5)のランドマークを取得
        wrist = hand[0]
        index_base = hand[5]

        # x, yの回転移動量を計算
        move_x = index_base.x - wrist.x
        move_y = index_base.y - wrist.y
        move_z = index_base.z - wrist.z

        movements.append((move_x, move_y, move_z))

    return movements

def fix_finger_movement(hand_landmarks, movements):
    if not hand_landmarks or not movements:
        return hand_landmarks  # 手が検出されなかった場合

    fixed_landmarks = []

    for hand_idx, hand in enumerate(hand_landmarks):
        movement = movements[hand_idx]
        fixed_hand = []
        for landmark in hand:
            fixed_landmark = type(landmark)(
                x=landmark.x - movement[0],
                y=landmark.y - movement[1],
                z=landmark.z - movement[2],
                visibility=landmark.visibility,
                presence=landmark.presence
            )
            fixed_hand.append(fixed_landmark)
        fixed_landmarks.append(fixed_hand)

    return fixed_landmarks

# 押したことの検出=指の付け根から先端までのランドマークで、z軸方向の変化を見る
def is_finger_pressed(hand_landmarks, finger_index, z_threshold=0.02):
    if not hand_landmarks:
        return False  # 手が検出されなかった場合

    for hand in hand_landmarks:
        base_landmark = hand[finger_index * 4 + 1]  # 指の付け根
        tip_landmark = hand[finger_index * 4 + 4]   # 指の先端

        # z軸方向の変化を計算
        z_diff = tip_landmark.z - base_landmark.z

        if z_diff < -z_threshold:
            return True  # 押されたと判断

    return False
