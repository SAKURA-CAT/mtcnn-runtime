#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2023-12-07 14:24:26
@File: test/test_mtcnn.py
@IDE: vscode
@Description:
    测试文件, 在此处测试mtcnn的效果
"""
import cv2
from PIL import Image
from mtcnn_rn import detect_faces


img = Image.open("test/imgs/3454d53f528b42dcab939a8f576da396.jpeg")

faces_info = detect_faces(img)
img = cv2.imread("test/imgs/3454d53f528b42dcab939a8f576da396.jpeg")


def draw_box(img, boxes):
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)


draw_box(img, faces_info[0])
