# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 21:13
# @Author  : Ziqi Wang
# @FileName: extract_img.py
# @Email: zw280@scarletmail.rutgers.edu

import read_txt, read_video

import os

path = os.path.abspath(os.path.dirname(__file__))
# print(path)
dataset = "\ets_dataset_concordia1_v1"
name_sample = "period1-1-2-gray"
txt_suffix = "-label.txt"
video_suffix = ".avi"

folder = os.path.exists(path + "\data")
if not folder:
    os.mkdir(path + "\data")
l = 0
for i in range(1, 4):
    for j in range(1, 3):
        for k in range(1, 7):
            name = "period{}-{}-{}-gray".format(i, j, k)
            lines = read_txt.read_txt(base_path=path + dataset, file_name=name + txt_suffix)
            # for l in range(len(lines)):
            #    with open(path + "\data" + "\\" + name + str(l) + txt_suffix, "w") as file:
            #       file.write(lines[l])
            im_folder = os.path.exists(path + "\data" + "\\" + name)
            with open(path + "\data" + "\\" + str(l) + ".txt", "w") as file:
                for item in lines:
                    file.write(item)
            if not im_folder:
                os.mkdir(path + "\data" + "\\" + str(l))
            read_video.video_to_img(base_path=path + dataset, file_name=name + video_suffix,
                                    img_path=path + "\data" + "\\" + str(l))
            l = l + 1

            print(name)
