# utils.py - ユーティリティ関数

import time
import cv2
import os
import urllib.request
import urllib.error

class CvFpsCalc:
    """FPS計算クラス"""

    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = [self._freq] * buffer_len

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)
        self._difftimes = self._difftimes[-10:]  # 最新10個保持

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps = round(fps, 2)

        return fps

def download_file(url, save_path):
    """ファイルをダウンロードする"""
    try:
        with urllib.request.urlopen(url) as web_file:
            with open(save_path, 'wb') as local_file:
                local_file.write(web_file.read())
    except urllib.error.URLError as e:
        print(e)