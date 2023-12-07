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
from mtcnnruntime import MTCNN
from PIL import Image, ImageDraw
from mtcnn_rn import detect_faces


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    if not isinstance(img_copy, Image.Image):
        img_copy = Image.fromarray(img_copy)
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="white")

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([(p[i] - 1.0, p[i + 5] - 1.0), (p[i] + 1.0, p[i + 5] + 1.0)], outline="blue")

    return img_copy


mtcnn = MTCNN()
path = "imgs/Solvay_conference_1927_comp.jpg"
img = cv2.imread(path)
boxes, landmarks = mtcnn.detect(img)
# img1 = Image.open(path)
# boxes, landmarks = detect_faces(img)
print(len(boxes), boxes, landmarks)

# 画出人脸框
img2 = show_bboxes(img, boxes, landmarks)
print(landmarks)
