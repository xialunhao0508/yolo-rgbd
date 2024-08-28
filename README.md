# yolo_rgb_SDK

## **1. 项目介绍**

yolo开源的视觉模型，只采用了rgb的纹理特征信息，对一些颜色纹理信息不明显的场景很难做到高适配性，我们的模型在采用rgb信息的同时，也考虑到了物体的深度信息，更好的增加了模型的泛化能力。基于此模型我们可以训练自己的权重，
来为特定场景下的实时目标检测而服务。在睿眼中， 我们可以用自己采集并标注好的训练集来训练特定的模型，来实现目标检测。我们现在的sdk提供了best.pt,
10500_50.pt两个权重。best.pt是一个比较通用的权重，而10500_50.pt是针对货架场景下的一些商品训练出来的权重。

- **API链接**：[API链接地址](http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/camera)

## **2. 代码结构**

```
yolorgb/
│
├── README.md        <- 项目的核心文档
├── requirements.txt    <- 项目的依赖列表
├── setup.py        <- 项目的安装脚本
├── .gitignore        <- 忽略文件
│
├── ultralytics/          <- 项目底层网络配置
├── yolo_rgbd/          <- 推理配置
│  ├── base.py       <- 基类
│  ├── general.py       <- 后处理逻辑
│  ├── interface.py       <- 推理接口
│  ├── network.py       <- 底层网络基类
│  └── solver.py/        <- 深度修复网络底层
└── tests/     <-  功能测试
```

## **3.环境与依赖**

* python3.8+
* opencv-python
* torchvision
* numpy
* pillow
* requests
* psutil
* pandas
* matplotlib
* pyyaml
* tqdm
* scipy

## **4. 安装说明**

1. 安装Python 3.8或者更高版本
2. 克隆项目到本地：`git clone http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/yolorgb.git`
3. 进入项目目录：`cd yolorgb`
4. 安装依赖：`pip install -r requirements.txt`
5. 编译打包：在与 `setup.py `文件相同的目录下执行以下命令：`python setup.py bdist_wheel`。 在 `dist` 文件夹中找到 `.wheel`
   文件，例如：`dist/yolorgb-0.1.0-py3-none-any.whl`。
6. 安装：`pip install yolorgb-0.1.0-py3-none-any.whl`

## **5. 使用指南**

### 推荐硬件&软件&环境配制

- 建议使用python3.8及以上版本。
- 3090Ti对应安装nvidia-driver-535
- 安装对应cuda版本11.7
- 安装对应torch和torchvision版本

### 权重描述

- yolo_weights：根据你自己本身的数据训练出来的模型文件,如:best.pt、10500_50.pt
- solver_weights：进行深度修复的权重, 如:CDNet.pth

### 权重下载

- yolo_weights：https://alidocs.dingtalk.com/i/nodes/QOG9lyrgJPReMyzGIMDX01LGWzN67Mw4?utm_scene=team_space
- solver_weights：https://alidocs.dingtalk.com/i/nodes/QG53mjyd80ZnAwpOtGOl0lMqV6zbX04v?utm_scene=team_space

## 6. 接口示例

```python
import cv2
from camera import realsense
from yolo_rgbd.solver import Solver
from yolo_rgbd.interface import YoloRGBD


def main(conf, input):
    # 初始化深度修复对象
    solver = Solver()
    # 初始化相机对象
    camera = realsense.RealSenseCamera()
    # 启动相机
    camera.start_camera()
    # 初始化yolo_rgbd对象
    rgbd = YoloRGBD()
    # yolo_rgbd 权重路径
    yolo_weights = r'best.pt'
    # 深度修复权重路径
    solver_weights = r'CDNet.pth'

    # 初始化模型
    model, solver_weights = rgbd.gen_model(yolo_weights=yolo_weights, solver=solver,
                                           solver_weights_path=solver_weights)
    while True:
        # 获取实时视频流
        color_img, depth_img, _, point_cloud, depth_frame = camera.read_align_frame()
        # rgb通道转换
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        # 深度修复
        deep_data_depth_esi = solver.test(color_img)
        # 深度通道转换
        deep_data3 = cv2.cvtColor(deep_data_depth_esi, cv2.COLOR_GRAY2BGR)
        # 模型推理
        results = rgbd.detect(model, color_img, deep_data3, conf)
        # 模型结果后处理
        annotated_frame, obj_img, mapped_depth = rgbd.backward_handle_output(results, color_img, depth_img, input)
        # 显示结果
        cv2.imshow("annotated_frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # 输出物体坐标
        if obj_img is not None:
            print('obj_img', obj_img)



if __name__ == '__main__':
    # 置信度调整
    conf = 0.25
    # 输入物体名称
    input = None

    main(conf, input)


```

## 7. **许可证信息**

说明项目的开源许可证类型（如MIT、Apache 2.0等）。

* 本项目遵循MIT许可证。

## 8. 常见问题解答（FAQ）**

列出一些常见问题和解决方案。

- **Q1：机械臂连接失败**

  答案：修改过机械臂IP

- **Q2：UDP数据推送接口收不到数据**

  答案：检查线程模式、是否使能推送数据、IP以及防火墙
