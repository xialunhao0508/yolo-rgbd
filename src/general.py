import cv2
import numpy as np
from numpy import ndarray


def label_backward_handle_output2(output: list, color_img: ndarray):
    """
    :param output：      模型直接返回的未处理的结果
    :param color_frame:     rgb彩色视频帧
    :param depth_frame:     d深度视频帧
    :param input:    需要检测的类别
    :return
        annotated_frame: 带有物体检测框的视频流
        obj_img: 当input为None时，返回所有物体的标签以及物体的center, box, wh, angle；input为固定类别时，返回对应类别的center, box, wh, angle，当检测画面中存在多个
                需要检测的类别时，先后顺序以检测置信度为准
        mapped_depth：mask对应的深度图
    """
    mapped_depth = None
    cls = []
    indices = []
    mapped_depth_list = []
    annotated_frame = output[0].plot()
    obj_img = []
    obj_img = np.empty((0, 4))
    if output[0].masks is not None:
        for cls_idx in range(len(output[0].boxes)):
            cls.append(output[0].names.get(int(output[0].boxes.cls[cls_idx].item())))
        for i, value in enumerate(cls):
            if value == input:
                indices.append(i)
        for i in range(len(indices)):
            mask = np.array(output[0].masks.data.cpu())[indices[i]]
            channel_zeros = mask == 0

            mapped_depth = np.zeros_like(color_img)
            mapped_depth[~channel_zeros] = color_img[~channel_zeros]
            mapped_depth[channel_zeros] = 0
            mapped_depth_list.append(mapped_depth)
            mask_position = np.where(mask == 1)
            mask_position = np.column_stack(
                (mask_position[1], mask_position[0])
            )

            rect = cv2.minAreaRect(mask_position)
            center, wh, angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            center = list(map(int, center))
            wh = list(map(int, wh))

            obj_img_list = np.array([center, box, wh, angle], dtype=object)
            obj_img = np.vstack((obj_img, obj_img_list))

    return annotated_frame, obj_img, mapped_depth


def label_backward_handle_output(outputs: list, color_img: ndarray):
    """
    :param output：      模型直接返回的未处理的结果
    :param color_frame:     rgb彩色视频帧
    :param depth_frame:     d深度视频帧
    :param input:    需要检测的类别
    :return
        annotated_frame: 带有物体检测框的视频流
        obj_img: 当input为None时，返回所有物体的标签以及物体的center, box, wh, angle；input为固定类别时，返回对应类别的center, box, wh, angle，当检测画面中存在多个
                需要检测的类别时，先后顺序以检测置信度为准
        mapped_depth：mask对应的深度图
    """
    mapped_depth = None
    cls = []
    indices = []
    mapped_depth_list = []
    annotated_frame = output[0].plot()
    obj_img = []
    obj_img = np.empty((0, 4))


    return annotated_frame, names, xyxys, masks

def no_label_backward_handle_output(output: list, color_img: ndarray):
    """
    :param output：      模型直接返回的未处理的结果
    :param color_frame:     rgb彩色视频帧
    :param depth_frame:     d深度视频帧
    :param input:    需要检测的类别
    :return
        annotated_frame: 带有物体检测框的视频流
        obj_img: 当input为None时，返回所有物体的标签以及物体的center, box, wh, angle；input为固定类别时，返回对应类别的center, box, wh, angle，当检测画面中存在多个
                需要检测的类别时，先后顺序以检测置信度为准
        mapped_depth：mask对应的深度图
    """
    mapped_depth = None
    cls = []
    indices = []
    mapped_depth_list = []
    annotated_frame = output[0].plot()
    obj_img = []
    obj_img = np.empty((0, 4))
    if output[0].masks is not None:
        obj_img = np.empty((0, 5))
        for cls_idx in range(len(output[0].boxes)):
            cls.append(output[0].names.get(int(output[0].boxes.cls[cls_idx].item())))
        for i, value in enumerate(cls):
            indices.append(i)
        for i in range(len(indices)):
            mask = np.array(output[0].masks.data.cpu())[indices[i]]
            channel_zeros = mask == 0

            mapped_depth = np.zeros_like(color_img)
            mapped_depth[~channel_zeros] = color_img[~channel_zeros]
            mapped_depth[channel_zeros] = 0
            mapped_depth_list.append(mapped_depth)
            mask_position = np.where(mask == 1)
            mask_position = np.column_stack(
                (mask_position[1], mask_position[0])
            )

            rect = cv2.minAreaRect(mask_position)
            center, wh, angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            center = list(map(int, center))
            wh = list(map(int, wh))

            obj_img_list = np.array([cls[i], center, box, wh, angle], dtype=object)
            obj_img = np.vstack((obj_img, obj_img_list))

    return annotated_frame, obj_img, mapped_depth
