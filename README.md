# mtcnn-runtime

基于mtcnn和onnxruntime的轻量级人脸检测库，开箱即用，非常方便。
![cover](./imgs/Solvay_conference_1927_comp_mtcnn-result.jpg)

## 使用方式

### 安装方式

在终端运行如下命令进行安装：

```shell
pip install mtcnn-runtime
```

> 本项目只针对python3.8以上版本进行维护

### 运行方式

安装完毕以后，可在python代码中加入如下内容进行使用：

```py
import cv2
from mtcnnruntime import MTCNN, draw_faces

mtcnn = MTCNN()
path="你的图像路径"
img = cv2.imread(path)
boxes, landmarks = mtcnn.detect(img)

# 并且, 你可以使用如下方式进行绘制
img_show = draw_faces(img, boxes, landmarks)
```
