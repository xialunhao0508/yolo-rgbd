from ultralytics import YOLO
from general import *
from base import DetectBase
from solver import Solver
import torch
from numpy import ndarray


class YoloRGBD(DetectBase):

    @staticmethod
    def forward_handle_input(color_frame: ndarray, depth_frame: ndarray):
        """
        :param color_frame：     rgb彩色视频帧
        :param depth_frame:     d深度视频帧
        :return:        rgb彩色视频帧、d深度视频帧
        """

        return color_frame, depth_frame

    @staticmethod
    def gen_model(yolo_weights: str, solver: Solver, solver_weights_path: str):
        """
        :param yolo_weights:        yolorgbd模型路径
        :param solver:      深度修复类对象，需要初始化模型
        :param solver_weights_path:     深度修复模型路径
        :return:        模型
        """
        model = YOLO(yolo_weights)
        solver.init_weights(solver_weights_path)
        return model, solver

    @staticmethod
    def backward_handle_output(outputs: list):
        """
        :param output：      模型直接返回的未处理的结果
        :param color_frame：     rgb彩色视频帧
        :param depth_frame:     d深度视频帧
        :param input:    需要检测的类别

        :return
        annotated_frame: 带有物体检测框的视频流
        obj_img: 当input为None时，返回所有物体的标签以及物体的center, box, wh, angle；input为固定类别时，返回对应类别的center, box, wh, angle，当检测画面中存在多个
                需要检测的类别时，先后顺序以检测置信度为准
        mapped_depth：mask对应的深度图
        """
        results = outputs[0]
        annotated_frame = results.plot()

        all_names = results.names

        # 获取结果
        masks = []
        names = []
        xyxys = []
        confs = []

        if results.masks:
            for _mask in results.masks:
                mask = (_mask.data * 255).to("cpu").to(torch.uint8).numpy()
                mask = np.squeeze(mask)
                masks.append(mask)
            for cls in results.boxes.cls:
                name = all_names.get(int(cls.item()))
                names.append(name)
            for xyxy in results.boxes.xyxy:
                xyxy = (int(xyxy[0].item()), int(xyxy[1].item())), (int(xyxy[2].item()), int(xyxy[3].item()))
                xyxys.append(xyxy)
            for conf in results.boxes.conf:
                conf = conf.item()
                confs.append(conf)
        return annotated_frame, names, xyxys, masks, confs

    @staticmethod
    def detect(model, color_img: ndarray, deep_data3: ndarray, conf: float = 0.7):
        """
        :param color_frame：     rgb彩色视频帧
        :param depth_frame:     d深度视频帧
        :param conf:            置信度阈值
        :return:        模型直接返回的未处理的结果
        """
        return model(color_img, deep_data3, conf=conf)

    @staticmethod
    def delete_model(model):
        """
        :param model： 模型
        """
        model.to('cpu')
        del model
        torch.cuda.empty_cache()
