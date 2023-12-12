from .detector import MTCNN


def draw_faces(img, boxes, landmarks):
    """画出人脸框"""
    import cv2

    img = img.copy()
    for box, landmark in zip(boxes, landmarks):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
        for i in range(5):
            cv2.circle(img, (int(landmark[i]), int(landmark[i + 5])), 2, (255, 0, 0), -1)
    return img


__all__ = ["MTCNN"]
