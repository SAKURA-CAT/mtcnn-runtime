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
from mtcnn_rn import detect_faces, MTCNN

mtcnn = MTCNN()


img1 = Image.open("test/imgs/Debra_Messing_0001.jpg")
img2 = cv2.imread("test/imgs/Debra_Messing_0001.jpg")
mtcnn.detect(image=img2)
