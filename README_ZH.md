# Sketching_Agent 项目文档

## 项目介绍

`Sketching_Agent` 是一个与强化学习（RL）相关的项目，主要用于草图重绘任务。项目结合了深度强化学习（DRL）和渲染技术，旨在使用更简洁的笔画实现高质量的草图重构任务。

## 使用说明

### 环境准备

* [PyTorch](http://pytorch.org/) 1.1.0
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)
* [opencv-python](https://pypi.org/project/opencv-python/) 3.4.0

```
pip3 install torch==1.1.0
pip3 install tensorboardX
pip3 install opencv-python
```

### 模型训练

在训练Sketching_Agent模型之前，需要先准备好渲染器和数据集。

#### 准备数据集

本项目使用了两个公开数据集：**MNIST** 和 **QuickDraw**。请按照以下说明从官网手动下载并放置到指定目录。

---

##### 🟦 MNIST 数据集下载

MNIST 是一个经典的手写数字识别数据集，包含 0~9 共 10 类数字的图像。

##### 🔗 下载地址：

请从 Yann LeCun 官方网站下载以下四个文件：
👉 http://yann.lecun.com/exdb/mnist/

##### 📄 所需文件：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

---

##### 🟨 QuickDraw 数据集下载

QuickDraw 是由 Google 开源的手绘草图数据集，包含 345 个类别，每个类别为 `.ndjson` 文件格式，记录笔画向量信息。

##### 🔗 下载地址：

👉 https://storage.googleapis.com/quickdraw_dataset/full/simplified/

例如下载 “cat” 类别的 `.ndjson` 文件：

```bash
wget https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson -P data/quickdraw/
```

#### 训练代码

训练渲染器模型：

```bash
python train_renderer.py
```

使用带有拉格朗日算子的DDPG方法训练Sketching_Agent：

```bash
python train_ddpg.py
```

### 模型测试

使用 `test_*.py` 文件进行模型测试。
例如测试quickdraw数据集

```bash
python test_quickdraw_128.py
```

#### 可视化结果

 `visualization_result` 模块中的工具来可视化模型生成的结果，并生成动图。
