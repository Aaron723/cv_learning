# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 20:36
# @Author  : Ziqi Wang
# @FileName: read_video.py
# @Email: zw280@scarletmail.rutgers.edu
import cv2

base_path = "D:\Pyproject\image_process\ets_dataset_concordia1_v1"
file_name = "period1-1-1-gray.avi"

data_path = "D:\Pyproject\image_process\data"


def video_to_img(base_path="D:\Pyproject\image_process\ets_dataset_concordia1_v1", file_name="period1-1-1-gray.avi",
                 img_path=None):
    file_path = base_path + "\\" + file_name
    video_capture = cv2.VideoCapture()
    video_capture.open(file_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # fps 是帧率，frame是一段视频中总的图片数量
    print("fps=", fps, "frames=", frames)
    for i in range(int(frames)):
        ret, frame = video_capture.read()
        file_name = file_name.split(".")[0]
        cv2.imwrite(img_path + "\\" + str(i) + ".png", frame)

# video_to_img()
