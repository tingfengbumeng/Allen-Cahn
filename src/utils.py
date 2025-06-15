"""
目的：实现Allen-Cahn方程解的可视化和L2误差评估。
方法：使用Matplotlib绘制神经网络预测结果的散点图和时间截面图，计算模型预测与真实标签之间的L2误差，支持批量预测以优化计算效率。
"""

import time
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from mindspore import Tensor
from mindspore import dtype as mstype


def visual(model, epochs=1, resolution=100, t_cross_sections=None, path=None):
    """
    可视化Allen-Cahn方程的神经网络预测结果。
    参数：
        model: 训练好的神经网络模型。
        epochs: 当前训练轮次，用于保存图像文件名。
        resolution: 时间和空间网格的分辨率，默认100。
        t_cross_sections: 时间截面列表，用于绘制特定时间的解曲线，默认为[0.25, 0.5, 0.75]。
    """
    # 生成时间和空间的均匀网格点
    t_flat = np.linspace(0, 1, resolution)
    x_flat = np.linspace(-1, 1, resolution)
    t_grid, x_grid = np.meshgrid(t_flat, x_flat)

    x = x_grid.reshape((-1, 1))
    t = t_grid.reshape((-1, 1))
    xt = Tensor(np.concatenate((x, t), axis=1), dtype=mstype.float32)

    # 使用模型预测 u(t, x)
    u_predict = model(xt)
    u_predict = u_predict.asnumpy()

    # 绘制完整 t-x 平面的散点图
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.scatter(t, x, c=u_predict, cmap=plt.cm.bwr)
    plt.xlabel('t')  # 设置 x 轴标签为时间
    plt.ylabel('x')  # 设置 y 轴标签为空间
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)

    # 设置默认时间截面
    if t_cross_sections is None:
        t_cross_sections = [0.25, 0.5, 0.75]

    # 绘制指定时间截面的 u(x) 曲线
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        xt = Tensor(
            np.stack([x_flat, np.full(x_flat.shape, t_cs)], axis=-1), dtype=mstype.float32)
        u = model(xt).asnumpy()
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')

    plt.tight_layout()

    if not path:
        plt.savefig(f'images/result.jpg')
    else:
        plt.savefig(path)


def _calculate_error(label, prediction):
    """
    计算预测值与真实标签之间的L2误差。
    """
    error = label - prediction
    l2_error = np.sqrt(
        np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))

    return l2_error


def _get_prediction(model, inputs, label_shape, batch_size):
    """
    根据输入数据计算模型预测结果。
    """
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()

    # 分批预测以避免内存溢出
    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print("    predict total time: {} ms".format((time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_l2_error(model, inputs, label, batch_size):
    """
    评估模型预测的L2误差。
    """
    label_shape = label.shape
    # 获取模型预测结果
    prediction = _get_prediction(model, inputs, label_shape, batch_size)
    label = label.reshape((-1, label_shape[1]))
    # 计算L2误差
    l2_error = _calculate_error(label, prediction)
    print("    l2_error: ", l2_error)
    print("=======================================================================================")
