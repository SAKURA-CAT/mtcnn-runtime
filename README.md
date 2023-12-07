# mtcnn-runtime
基于mtcnn和onnxruntime的轻量级人脸检测库，开箱即用，非常方便。

# 使用方式
## 安装方式
在终端运行如下命令进行安装：
```shell
pip install mtcnn-runtime
```
> 本项目只针对python3.8以上版本进行维护
## 运行方式
安装完毕以后，可在python代码中加入如下内容进行使用：
```py
import cv2
from mtcnnruntime import MTCNN

mtcnn = MTCNN()
path="你的图像路径"
img = cv2.imread(path)
boxes, landmarks = mtcnn.detect(img)
```
