"""
目的：定义Allen-Cahn方程的PDE损失计算和多尺度全连接神经网络模型。
方法：通过继承PDEWithLoss类实现Allen-Cahn方程的PDE定义和损失计算，使用SymPy定义方程并转换为MindSpore节点；通过继承MultiScaleFCSequential类实现带输出变换的多尺度全连接神经网络，支持自定义输出变换函数。
"""

import numpy as np
from sympy import diff, symbols, Function

from mindspore import ops, Tensor
from mindspore import dtype as mstype

from mindflow.pde import PDEWithLoss, sympy_to_mindspore
from mindflow.loss import get_loss_metric
from mindflow.cell import MultiScaleFCSequential


class AllenCahn(PDEWithLoss):
    """
    Allen-Cahn方程的PDE损失类，用于定义方程并计算物理驱动的神经网络损失。
    继承自PDEWithLoss，通过SymPy定义PDE并转换为MindSpore计算节点，支持输出变换以满足边界条件。
    """

    def __init__(self, model, loss_fn="mse"):
        """
        初始化Allen-Cahn方程的PDE模型。
        参数：
            model: 神经网络模型。
            loss_fn: 损失函数，默认为均方误差（mse），可传入字符串或自定义函数。
        """
        # 定义符号变量 x（空间）和 t（时间），以及函数 u(x, t)
        self.x, self.t = symbols("x t")
        self.u = Function("u")(self.x, self.t)
        self.in_vars = [self.x, self.t]
        self.out_vars = [self.u]
        # 根据输入设置损失函数，字符串则使用预定义损失函数
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        # 将PDE方程转换为MindSpore计算节点
        self.pde_nodes = sympy_to_mindspore(
            self.pde(), self.in_vars, self.out_vars)
        # 设置模型的输出变换函数以满足边界条件
        model.set_output_transform(self.output_transform)
        super(AllenCahn, self).__init__(model, self.in_vars, self.out_vars)

    def output_transform(self, x, out):
        """
        输出变换函数，调整神经网络输出以满足Allen-Cahn方程的边界条件。
        参数：
            x: 输入张量，包含空间坐标 x 和时间 t。
            out: 神经网络的原始输出。
        返回：
            调整后的输出，结合边界条件（如 x=±1 时值为零）。
        """
        return x[:, 0:1] ** 2 * ops.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * out

    def force_function(self, u):
        """
        定义Allen-Cahn方程的非线性力项。
        参数：
            u: 函数值 u(x, t)。
        返回：
            非线性项 5 * (u - u^3)。
        """
        return 5 * (u - u ** 3)

    def pde(self):
        """
        定义Allen-Cahn偏微分方程。
        返回：
            包含PDE损失的字典，形式为 u_t - d * u_xx - f(u)，其中 d=0.001。
        """
        d = 0.001
        loss_1 = (
            self.u.diff(self.t)  # u 对时间 t 的一阶导数
            - d * diff(self.u, (self.x, 2))  # 负 d 乘以 u 对 x 的二阶导数
            - self.force_function(self.u)  # 减去非线性力项
        )
        return {"loss_1": loss_1}

    def get_loss(self, pde_data):
        """
        计算PDE残差损失。
        参数：
            pde_data: 输入数据，包含空间和时间坐标。
        返回：
            PDE损失值，基于PDE残差与零的均方误差。
        """
        pde_res = ops.Concat(1)(self.parse_node(
            self.pde_nodes, inputs=pde_data))  # 拼接PDE节点计算结果
        pde_loss = self.loss_fn(
            pde_res, Tensor(np.array([0.0]).astype(
                np.float32), mstype.float32)  # 计算与零的损失
        )

        return pde_loss


class MultiScaleFCSequentialOutputTransform(MultiScaleFCSequential):
    """
    多尺度全连接神经网络类，支持输出变换功能。
    继承自MultiScaleFCSequential，通过多尺度分支处理输入，支持潜在向量和自定义输出变换，增强对复杂特征的建模。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 weight_norm=False,
                 has_bias=True,
                 bias_init="default",
                 num_scales=4,
                 amp_factor=1.0,
                 scale_factor=2.0,
                 input_scale=None,
                 input_center=None,
                 latent_vector=None,
                 output_transform=None
                 ):
        # 初始化父类并设置输出变换
        super(MultiScaleFCSequentialOutputTransform, self).__init__(in_channels, out_channels,
                                                                    layers, neurons, residual,
                                                                    act, weight_init,
                                                                    weight_norm, has_bias,
                                                                    bias_init, num_scales,
                                                                    amp_factor, scale_factor,
                                                                    input_scale, input_center,
                                                                    latent_vector)
        self.output_transform = output_transform

    def set_output_transform(self, output_transform):
        """设置输出变换函数，用于调整网络输出以满足特定约束"""
        self.output_transform = output_transform

    def construct(self, x):
        """
        构建多尺度网络前向传播。
        参数：
            x: 输入张量。
        返回：
            多尺度网络输出，若设置了输出变换则应用变换函数。
        """
        x = self.input_scale(x)  # 对输入进行缩放
        if self.latent_vector is not None:
            # 若存在潜在向量，调整形状并与输入拼接
            batch_size = x.shape[0]
            latent_vectors = self.latent_vector.view(
                self.num_scenarios, 1, self.latent_size)
            latent_vectors = latent_vectors.repeat(batch_size // self.num_scenarios,
                                                   axis=1).view((-1, self.latent_size))
            x = self.concat((x, latent_vectors))
        out = 0
        # 多尺度处理：各分支使用不同缩放系数处理输入并累加输出
        for i in range(self.num_scales):
            # 缩放输入以捕捉不同尺度特征
            x_s = x * self.scale_coef[i]
            # 计算并累加分支输出
            out = out + self.cast(self.cell_list[i](x_s), mstype.float32)
        if self.output_transform is None:
            return out

        # 应用输出变换函数，调整输出以满足问题约束
        return self.output_transform(x, out)
