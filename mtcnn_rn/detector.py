import numpy as np
from .first_stage import run_first_stage
import onnxruntime
import os
import numpy as np
import math
import cv2
from typing import Tuple

from PIL import Image


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

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

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
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        img = np.asarray(image, "float32")
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


def detect_faces(
    image: np.ndarray, min_face_size: float = 20.0, thresholds: list = [0.7, 0.7, 0.7], nms_thresholds: list = [0.6, 0.7, 0.8]
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

    # ---------------------------------- 加载模型 ----------------------------------

    basedir = os.path.dirname(__file__)
    providers = ["CPUExecutionProvider"]
    pnet = onnxruntime.InferenceSession(f"{basedir}/weights/pnet.onnx", providers=providers)  # Load a ONNX model
    input_name_pnet = pnet.get_inputs()[0].name
    output_name_pnet1 = pnet.get_outputs()[0].name
    output_name_pnet2 = pnet.get_outputs()[1].name
    pnet = [pnet, input_name_pnet, [output_name_pnet1, output_name_pnet2]]

    rnet = onnxruntime.InferenceSession(f"{basedir}/weights/rnet.onnx", providers=providers)  # Load a ONNX model
    input_name_rnet = rnet.get_inputs()[0].name
    output_name_rnet1 = rnet.get_outputs()[0].name
    output_name_rnet2 = rnet.get_outputs()[1].name
    rnet = [rnet, input_name_rnet, [output_name_rnet1, output_name_rnet2]]

    onet = onnxruntime.InferenceSession(f"{basedir}/weights/onet.onnx", providers=providers)  # Load a ONNX model
    input_name_onet = onet.get_inputs()[0].name
    output_name_onet1 = onet.get_outputs()[0].name
    output_name_onet2 = onet.get_outputs()[1].name
    output_name_onet3 = onet.get_outputs()[2].name
    onet = [onet, input_name_onet, [output_name_onet1, output_name_onet2, output_name_onet3]]

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size=24)

    output = rnet[0].run([rnet[2][0], rnet[2][1]], {rnet[1]: img_boxes})
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

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    # img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    # with torch.no_grad():
    #     img_boxes = torch.FloatTensor(img_boxes)
    # output = onet(img_boxes)
    output = onet[0].run([onet[2][0], onet[2][1], onet[2][2]], {rnet[1]: img_boxes})
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


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].

    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
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
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].

    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image.

    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.

    Returns:
        a float numpy array of shape [n, 3, size, size].
    """

    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")

        img_array = np.asarray(img, "uint8")
        img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
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
