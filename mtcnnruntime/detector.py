import numpy as np
import onnxruntime
import os
import numpy as np
import cv2
from typing import Tuple
import math


class MTCNN(object):
    __BASE_DIR = os.path.join(os.path.dirname(__file__), "weights")
    __PNET = os.path.join(__BASE_DIR, "pnet.onnx")
    __RNET = os.path.join(__BASE_DIR, "rnet.onnx")
    __ONET = os.path.join(__BASE_DIR, "onet.onnx")

    def __init__(self) -> None:
        """初始化MTCNN模型,加载权重,"""
        providers = ["CPUExecutionProvider"]
        pnet = onnxruntime.InferenceSession(self.__PNET, providers=providers)
        rnet = onnxruntime.InferenceSession(self.__RNET, providers=providers)
        onet = onnxruntime.InferenceSession(self.__ONET, providers=providers)

        # ---------------------------------- 加载pnent ----------------------------------

        input_name_pnet = pnet.get_inputs()[0].name
        output_name_pnet1 = pnet.get_outputs()[0].name
        output_name_pnet2 = pnet.get_outputs()[1].name
        self.__pnet = [pnet, input_name_pnet, [output_name_pnet1, output_name_pnet2]]

        # ---------------------------------- 加载rnet ----------------------------------

        input_name_rnet = rnet.get_inputs()[0].name
        output_name_rnet1 = rnet.get_outputs()[0].name
        output_name_rnet2 = rnet.get_outputs()[1].name
        self.__rnet = [rnet, input_name_rnet, [output_name_rnet1, output_name_rnet2]]

        # ---------------------------------- 加载onet ----------------------------------

        input_name_onet = onet.get_inputs()[0].name
        output_name_onet1 = onet.get_outputs()[0].name
        output_name_onet2 = onet.get_outputs()[1].name
        output_name_onet3 = onet.get_outputs()[2].name
        self.__onet = [onet, input_name_onet, [output_name_onet1, output_name_onet2, output_name_onet3]]

    def detect(
        self, image: np.ndarray, min_face_size: float = 20.0, thresholds: list = [0.7, 0.7, 0.7], nms_thresholds: list = [0.6, 0.7, 0.8]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """检测图中的人像，并为它们返回边界框和点。

        Parameters
        ----------
        image : numpy.ndarray
            输入图像,可以为BGRA、BGR像素通道或单像素通道
        min_face_size : float, optional
            最小的人脸大小，这意味着小于此阈值的人脸将被丢弃, 默认为 20.0
        thresholds : list, optional
            阈值列表，用于三个网络的置信度阈值, 默认为 [0.7, 0.7, 0.7]
        nms_thresholds : list, optional
            非极大值抑制阈值列表，用于三个网络的非极大值抑制阈值, 默认为 [0.6, 0.7, 0.8]

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            返回一个元组，第一个元素为边界框信息，第二个元素为关键点点信息
        """

        # ---------------------------------- 建立图像金字塔 ----------------------------------
        width, height = image.shape[:2]
        min_length = min(height, width)
        # 金字塔的最小尺寸
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        # 存储图像金字塔的尺度缩放值
        scales = []
        # 为了使我们能够检测到的最小尺寸等于我们想要检测到的最小人脸尺寸，因此将图像缩放到的最小尺寸
        m = min_detection_size / min_face_size
        min_length *= m
        # 计算图像金字塔的尺度缩放值
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        # ---------------------------------- STAGE 1 ----------------------------------

        # 边界框信息，将会被返回
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = self.__run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # 收集来自不同尺度的边界框(和偏移量、分数)
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # 使用 pnet 预测的偏移量来转换边界框
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # ---------------------------------- STAGE2 ----------------------------------

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        output = self.__rnet[0].run([self.__rnet[2][0], self.__rnet[2][1]], {self.__rnet[1]: img_boxes})
        offsets = output[0]  # shape [n_boxes, 4]
        probs = output[1]  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # ---------------------------------- STAGE 3 ----------------------------------

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        # img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        # with torch.no_grad():
        #     img_boxes = torch.FloatTensor(img_boxes)
        # output = onet(img_boxes)
        output = self.__onet[0].run([self.__onet[2][0], self.__onet[2][1], self.__onet[2][2]], {self.__rnet[1]: img_boxes})
        landmarks = output[0]  # shape [n_boxes, 10]
        offsets = output[1]  # shape [n_boxes, 4]
        probs = output[2]  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def __run_first_stage(self, image: np.ndarray, scale: float, threshold: float) -> np.ndarray:
        """运行第一阶段的网络，也就是pnet

        Parameters
        ----------
        image : np.ndarray
            输入图像
        scale : float
            缩放比例
        threshold : float
            在从网络的预测生成边界框时，对人脸概率的阈值。

        Returns
        -------
        np.ndarray
            形状为 [n_boxes, 9] 的浮点数 numpy 数组，包含得分和偏移的边界框（4 + 1 + 4）。
        """
        # 使用 OpenCV 将图像缩放
        width, height = image.shape[:2]
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = cv2.resize(image, (sh, sw), cv2.INTER_LINEAR)
        img = np.asarray(img, "float32")
        img = _preprocess(img)
        output = self.__pnet[0].run([self.__pnet[2][0], self.__pnet[2][1]], {self.__pnet[1]: img})
        #  在每个滑动窗口上检测到人脸的概率
        probs = output[1][0, 1, :, :]
        # 转换为真实边界框的变换
        offsets = output[0]
        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]


# ---------------------------------- 工具函数 ----------------------------------


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack(
        [
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score,
            offsets,
        ]
    )
    # why one is added?

    return bounding_boxes.T


def nms(boxes, overlap_threshold=0.5, mode="union"):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:
        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == "union":
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]]))

    return pick


def convert_to_square(bboxes: np.ndarray) -> np.ndarray:
    """将边界框转换为正方形形式。

    参数：
        bboxes：形状为 [n, 5] 的浮点数 numpy 数组。

    返回：
        形状为 [n, 5] 的浮点数 numpy 数组，
            转换为正方形的边界框。
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    square_bboxes[:, 4] = bboxes[:, 4]  # 保留第五列（例如，分数或类别信息）
    return square_bboxes


def calibrate_box(bboxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """将边界框校准为真实边界框，使用网络输出的“offsets”。

    Parameters
    ----------
    bboxes : np.ndarray
        形状为 (n, 4) 的边界框数组，包含 x1, y1, x2, y2 的坐标。
    offsets : np.ndarray
        形状为 (n, 4) 的偏移数组，用于校准边界框。

    Returns
    -------
    np.ndarray
        校准后的边界框数组，形状为 (n, 4)。
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    # 为了使得 x1 < x2 且 y1 < y2，我们需要将 w 和 h 转换为列向量
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # 这里的操作相当于以下代码：
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # 下面的写法更加紧凑

    # 注意：这里的偏移是否总是使得 x1 < x2 且 y1 < y2？

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes: np.ndarray, img: np.ndarray, size: int = 24) -> np.ndarray:
    """从图像中剪切出边界框。

    参数：
        bounding_boxes: 形状为 [n, 5] 的浮点数 numpy 数组。
        img: 形状为 [h, w, c] 的整数 numpy 数组。
        size: 整数，剪切图像的大小。

    返回：
        形状为 [n, 3, size, size] 的浮点数 numpy 数组。
    """
    num_boxes = len(bounding_boxes)
    width, height = img.shape[:2]

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")
        img_array = np.asarray(img, "uint8")
        try:
            img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :]
        except ValueError:
            pass
        # 调整大小
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_AREA)
        img_box = np.asarray(img_box, "float32")

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype("int32") for i in return_list]

    return return_list


def _preprocess(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img
