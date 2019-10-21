# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 20:12
# @Author  : Ziqi Wang
# @FileName: read_txt.py
# @Email: zw280@scarletmail.rutgers.edu


# base_path = "D:\Pyproject\image_process\ets_dataset_concordia1_v1"
# file_name = "period1-1-1-gray-label.txt"


def read_txt(base_path="F:\ets_dataset_concordia1_v1", file_name="period1-1-1-gray-label.txt"):
    lines = list()
    with open(base_path + "\\" + file_name) as file:
        strs = file.readlines()
        for line in strs:
         #   line = line[0:-1]
            lines.append(line)

        file.close()
    return lines


# print(len(read_txt()))
