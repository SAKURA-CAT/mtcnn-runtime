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
from mtcnnruntime import MTCNN, draw_faces


mtcnn = MTCNN()
path = "imgs/Solvay_conference_1927_comp.jpg"
img = cv2.imread(path, -1)
boxes, landmarks = mtcnn.detect(img)
# img = Image.open(path)
# img = Image.fromarray(img)
# boxes, landmarks = detect_faces(img)
print(len(boxes), boxes, landmarks)

# 画出人脸框
img2 = draw_faces(img, boxes, landmarks)
print(landmarks)
