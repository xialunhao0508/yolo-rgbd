from setuptools import setup, find_packages

setup(
    name="yolo_rgb",
    version="0.1.0",
    description='yolo开源的视觉模型，只采用了rgb的纹理特征信息，对一些颜色纹理信息不明显的场景很难做到高适配性，我们的模型在采用rgb信息的同时，也考虑到了物体的深度信息，更好的增加了模型的泛化能力；',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/yolorgb.git",
    packages=find_packages(exclude=['tests']),
    # author='Your Name',  # 作者名称
    # author_email='your@email.com',  # 作者邮箱
    # license="MIT", # 许可证
    install_requires=["opencv-python", "pyrealsense2", "onnxruntime", "matplotlib", "Pyyaml", "requests", "tqdm",
                      "psutil", "pandas"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)
