"""
目的：为Allen-Cahn方程的训练和测试创建数据集。
方法：使用MindFlow和SciPy库，基于配置文件生成时间-空间域的训练数据集，加载MAT文件生成测试数据集，处理为MindSpore Tensor格式。
"""

import os
import numpy as np

from scipy.io import loadmat

from mindspore import Tensor
from mindspore import dtype as mstype

from mindflow.data import Dataset
from mindflow.geometry import Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import generate_sampling_config


def create_training_dataset(config):
    # 获取几何和数据配置
    geom_config = config["geometry"]
    data_config = config["data"]

    # 定义时间域和空间域
    time_interval = TimeDomain(
        "time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Interval(
        "domain", geom_config["coord_min"], geom_config["coord_max"])
    region = GeometryWithTime(spatial_region, time_interval)
    # 设置采样配置
    region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {region: ["domain"]}
    # 初始化数据集
    dataset = Dataset(geom_dict)

    return dataset


def create_test_dataset(test_dataset_path):
    # 加载测试数据集（Allen-Cahn数据集）
    test_data = loadmat(os.path.join(test_dataset_path, "Allen_Cahn.mat"))
    x, t, u = test_data["x"], test_data["t"], test_data["u"]
    xx, tt = np.meshgrid(x, t)

    # 将网格数据展平并组合为测试数据，转换为MindSpore的Tensor格式
    test_data = Tensor(np.vstack((np.ravel(xx), np.ravel(tt))
                                 ).T.astype(np.float32), mstype.float32)
    # 将标签u展平为一维数组
    test_label = u.flatten()[:, None]
    return test_data, test_label
