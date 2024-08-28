# Introduction:
yolo开源的视觉模型，只采用了rgb的纹理特征信息，对一些颜色纹理信息不明显的场景很难做到高适配性，我们的模型在采用rgb信息的同时，也考虑到了物体的深度信息，更好的增加了模型的泛化能力；

# Prerequirements
## Recommended Hardware & Software & Environment
3090Ti对应安装nvidia-driver-535<br>
安装对应cuda版本11.7<br>
安装对应torch和torchvision版本<br>
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

## Weights Description：
1、yolo_weights：根据你自己本身的数据训练出来的模型文件,如:best.pt、10500_50.pt<br>
2、solver_weights：进行深度修复的权重, 如:CDNet.pth

## Weights Download
1、yolo_weights：https://alidocs.dingtalk.com/i/nodes/QOG9lyrgJPReMyzGIMDX01LGWzN67Mw4?utm_scene=team_space<br>
2、solver_weights：https://alidocs.dingtalk.com/i/nodes/QG53mjyd80ZnAwpOtGOl0lMqV6zbX04v?utm_scene=team_space

# Distribution
1. Execute the following commands in the same directory as setup.py: `python setup.py bdist_wheel`
2. find the `.wheel` in folder `dist`, example: `dist/vertical_grab-0.1.0-py3-none-any.whl`

# Install

`pip install vertical_grab-0.1.0-py3-none-any.whl`


# Unit Test
1. cd into `tests`
2. execute: `pytest`
