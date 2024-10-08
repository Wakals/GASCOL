import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from adan import Adan
from scipy.stats import norm

from PIL import Image

import torch
from torch import nn

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distCUDA2

from sh_utils import eval_sh, SH2RGB, RGB2SH

softmax = nn.Softmax(dim=1)

# 定义了四元数乘积
def quaternion_multiply(q, p):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = p.unbind(dim=-1)
    return torch.stack([
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0], dim=-1)

# inverse activation
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 调整学习率函数
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

# 取下三角矩阵的元素
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

"""
计算三维高斯分布的权重系数, xyzs 是位置坐标, covs 是协方差矩阵的元素
"""
def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

# 构建旋转矩阵
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# 构建放缩矩阵
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# 储存点云基本信息
class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

# GS 模型
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, center=[0, 0, 0], edge=[1, 1, 1]):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)    # 颜色
        self._features_rest = torch.empty(0)     # 啥也没有
        self._segment_p = torch.empty(0)  # 3D Segmentation probability
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.child = []
        self.object_center = nn.Parameter(torch.tensor(np.asarray(center)).float().cuda().requires_grad_(True))
        self.scale_factor = nn.Parameter(torch.tensor(1).float().cuda().requires_grad_(True))
        self.object_edge = edge
        self.floor = None
        self.prompt = ""
        self.color = None
        self.ori = 0
        self.setup_functions()

        self.fix_xyz = torch.empty(0)
        self.other_xyz = torch.empty(0)
        self.fix_features_dc = torch.empty(0)
        self.other_features_dc = torch.empty(0)
        self.fix_features_rest = torch.empty(0)
        self.other_features_rest = torch.empty(0)
        self.fix_segment_p = torch.empty(0)
        self.other_segment_p = torch.empty(0)
        self.fix_scaling = torch.empty(0)
        self.other_scaling = torch.empty(0)
        self.fix_rotation = torch.empty(0)
        self.other_rotation = torch.empty(0)
        self.fix_opacity = torch.empty(0)
        self.other_opacity = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._segment_p,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._segment_p,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # 获取各个属性的方法
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scene_scaling(self):
        return torch.log(self.scaling_activation(self._scaling) * self.scale_factor)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scene_rotation(self):
        theta = self.ori * np.pi / 180
        q_y = torch.tensor([np.cos(theta / 2), 0, np.sin(theta / 2), 0], dtype=torch.float, device=self._rotation.device)
        rotation = quaternion_multiply(q_y, self._rotation)
        return rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_segment_p(self):
        return softmax(self._segment_p)
        # return self._segment_p
    
    # 加入平移尺缩旋转之后的位置
    @property
    def get_scene_xyz(self):
        theta = torch.tensor(self.ori, dtype=torch.float, device="cuda") * (np.pi / 180)
        rotation_matrix = torch.tensor([
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)]
        ], dtype=torch.float, device="cuda")
        return torch.mm(self._xyz, rotation_matrix.t()) * self.scale_factor + self.object_center

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # 增加当前活动的球谐度 (?)
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    # 計算当前协方差矩阵
    def get_covariance(self, scaling_modifier = 1.0):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 从给定点云数据创建高斯模型
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float = 1, floor : bool = False):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if floor:
            opacities = inverse_sigmoid(0.2 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 允许梯度回传
        if floor:
            self._xyz = nn.Parameter(fused_point_cloud).requires_grad_(False)
        else:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._segment_p = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda")).requires_grad_(True)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # if floor:
        #     self._opacity = nn.Parameter(opacities).requires_grad_(False)
        # else:
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, stage=1):
        # self.percent_dense = training_args.percent_dense
        # 用于存储每个点的梯度累积和分母
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 设置学习率的列表, 特别注意xyz的学习率设置方式
        if stage == 1:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * sum(self.object_edge) / len(self.object_edge), "name": "xyz"},
                {'params': [self._segment_p], 'lr': 0.001, "name": "segment_p"},  # set learning rate of segment_p to a fixed value
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self.object_center], 'lr': 0.00001, "name": "center"},
                {'params': [self.scale_factor], 'lr': 0.0000025, "name": "scale_factor"},
            ]
        elif stage == 2:
            l = [
                {'params': [self._xyz], 'lr': 0.0, "name": "xyz"},
                {'params': [self._segment_p], 'lr': 0.0015, "name": "segment_p"},  # set learning rate of segment_p to a fixed value
                {'params': [self._features_dc], 'lr': 0.0, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': 0.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': 0.0, "name": "opacity"},
                {'params': [self._scaling], 'lr': 0.0, "name": "scaling"},
                {'params': [self._rotation], 'lr': 0.0, "name": "rotation"},
                {'params': [self.object_center], 'lr': 0.0, "name": "center"},
                {'params': [self.scale_factor], 'lr': 0.0, "name": "scale_factor"},
            ]
        elif stage == 3:
            l = [
                {'params': [self.object_center], 'lr': 0.00001, "name": "center"},
                {'params': [self.scale_factor], 'lr': 0.0000025, "name": "scale_factor"},
                {'params': [self.other_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * sum(self.object_edge) / len(self.object_edge), "name": "other_xyz"},
                {'params': [self.other_segment_p], 'lr': 0.001, "name": "other_segment_p"},  # set learning rate of segment_p to a fixed value
                {'params': [self.other_features_dc], 'lr': training_args.feature_lr, "name": "other_f_dc"},
                {'params': [self.other_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "other_f_rest"},
                {'params': [self.other_opacity], 'lr': training_args.opacity_lr, "name": "other_opacity"},
                {'params': [self.other_scaling], 'lr': training_args.scaling_lr, "name": "other_scaling"},
                {'params': [self.other_rotation], 'lr': training_args.rotation_lr, "name": "other_rotation"},
            ]
        else:
            raise ValueError("Invalid stage number")

        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer = Adan(l, lr=0.0, eps=1e-15)
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param is self.other_features_dc:
                    print(f'\033[1;32;40m[INFO] other_features_dc correctly added to optimizer.\033[0m')
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * sum(self.object_edge) / len(self.object_edge),
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale * sum(self.object_edge) / len(self.object_edge),
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    # 重置球谐特征
    def reset_sh(self):
        if self.color:
            rgb = np.array(self.color) / 255
            color = np.tile(rgb, (self._xyz.shape[0], 1))
            fused_color = RGB2SH(torch.tensor(color).float().cuda())
        else:
            fused_color = RGB2SH(torch.tensor(SH2RGB(np.random.random((self._xyz.shape[0], 3)) / 255.0)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._segment_p = nn.Parameter(torch.zeros((self._features_dc.shape[0], 2), dtype=torch.float, device="cuda") * 0.5).requires_grad_(True)
        print(f'In reset_sh the shape of segment_p is {self._segment_p.shape}')

    # 在每个训练迭代步骤更新学习率
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr = self.xyz_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr
            elif param_group["name"] == "other_xyz":
                param_group['lr'] = lr
        #     if param_group["name"] == "f_dc" or param_group["name"] == "f_rest":
        #         param_group['lr'] = lr * 10
        # if iteration > 200:
        #     for param_group in self.optimizer.param_groups:
        #         if param_group["name"] == "xyz" or param_group["name"] == "opacity":
        #             param_group['lr'] = 0.0

    # 构建属性列表
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._segment_p.shape[1]):
            l.append('segment_p_{}'.format(i))
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # 保存模型
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        segment_p = self._segment_p.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, segment_p, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 重置透明度
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # 加载模型ckpt
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        print("Number of points at loading : ", xyz.shape[0])

        segment_p = np.zeros((xyz.shape[0], 2))
        segment_p[:, 0] = np.asarray(plydata.elements[0]["segment_p_0"])
        segment_p[:, 1] = np.asarray(plydata.elements[0]["segment_p_1"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._segment_p = nn.Parameter(torch.tensor(segment_p, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    # 将一个新的张量替换到优化器中，同时保留与该张量相关的优化状态
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # red warning to tell it is here
        print(f"\033[1;32;40m[INFO] I'm HERE!!!\033[0m")
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        print(f"\033[1;32;40m[INFO] I'm HERE!!!\033[0m")
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        print(f"\033[1;32;40m[INFO] I'm HERE!!!\033[0m")
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        print(f"\033[1;32;40m[INFO] I'm HERE!!!\033[0m")
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

# 透视投影矩阵
def getProjectionMatrix(znear, zfar, fovX, fovY):
    # 计算切线
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# 摄像机模型，用于在 3D 空间中进行渲染
class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        # c2w (pose) should be in NeRF convention.

        # 参数
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        # 世界到相机的变换矩阵
        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        # 计算投影矩阵
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()  # 相机中心

# 渲染场景
class Renderer:
    def __init__(self, sh_degree=3, white_background=True, radius=1, centers=None, edges=None, floor=False):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)
        for center, edge in zip(centers, edges):
            self.gaussians.child.append(GaussianModel(sh_degree, center, edge))
        if floor:
            self.gaussians.floor = GaussianModel(sh_degree)
        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        # load checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def initialize(self, input=None, num_pts=100000):
        if input is None:
            # red print to tell input is None
            # print(f"\033[1;31;40m[INFO] input is None\033[0m")
            # init from random point cloud
            for i, chi in enumerate(self.gaussians.child):
                # x = np.random.random((num_pts,)) * chi.object_edge[0] * 0.8 - chi.object_edge[0] * 0.4
                # y = np.random.random((num_pts,)) * chi.object_edge[1] * 0.8 - chi.object_edge[1] * 0.4
                # z = np.random.random((num_pts,)) * chi.object_edge[2] * 0.8 - chi.object_edge[2] * 0.4
                # xyz = np.stack((x, y, z), axis=1)

                r = norm.rvs(size=num_pts)
                eps = 1e-10
                distances = 0.5 / (r + eps)
                directions = np.random.normal(size=(num_pts, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
                object_edge = np.array(chi.object_edge)
                xyz = directions * distances[:, np.newaxis]
                xyz *= object_edge / 2
                xyz = np.clip(xyz, -object_edge / 2, object_edge / 2)

                shs = np.random.random((num_pts, 3)) / 255.0
                color = SH2RGB(shs)

                pcd = BasicPointCloud(
                    points=xyz, colors=color, normals=np.zeros((num_pts, 3))
                )
                chi.create_from_pcd(pcd, 10)
            # del self.xm, self.model, self.diffusion

            if self.gaussians.floor != None:
                # init the floor
                num_along_axis = 600
                num_pts = num_along_axis ** 2
                x = np.linspace(-0.9, 0.9, num=num_along_axis)
                z = np.linspace(-0.9, 0.9, num=num_along_axis)
                y = np.full((num_pts,), 0)
                x, z = np.meshgrid(x, z)
                x, z = x.flatten(), z.flatten()
                xyz = np.stack((x, y, z), axis=1)
                shs = np.random.random((num_pts, 3)) / 255.0
                color = SH2RGB(shs)

                pcd = BasicPointCloud(
                    points=xyz, colors=color, normals=np.zeros((num_pts, 3))
                )
                self.gaussians.floor.create_from_pcd(pcd, 10, floor=True)
        elif isinstance(input, BasicPointCloud):
            # print(f"\033[1;31;40m[INFO] input is BasicPointCloud\033[0m")
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # print(f"\033[1;31;40m[INFO] input is not None\033[0m")
            # load from saved ply
            self.gaussians.load_ply(input)

        # raise ValueError('stop here')

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        gs_model=None,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gs_model.get_xyz,
                dtype=gs_model.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gs_model.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs_model.get_xyz
        means2D = screenspace_points
        opacity = gs_model.get_opacity

        segment_p = gs_model.get_segment_p

        # print(f'the shape of segment_p in gs_model is {segment_p.shape}')

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = gs_model.get_covariance(scaling_modifier)
        else:
            scales = gs_model.get_scaling
            rotations = gs_model.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = gs_model.get_features.transpose(1, 2).view(
                    -1, 3, (gs_model.max_sh_degree + 1) ** 2
                )
                dir_pp = gs_model.get_xyz - viewpoint_camera.camera_center.repeat(
                    gs_model.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    gs_model.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = gs_model.get_features
        else:
            colors_precomp = override_color

        # print(f'the shape of shs is {shs.shape}')
        if colors_precomp is not None:
            print(f'the shape of color precomp is {colors_precomp.shape}')
        # raise ValueError('stop here')
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # segment_p = torch.cat((segment_p, torch.zeros((segment_p.shape[0], 1), device=segment_p.device)), dim=1)
        # segment_p = segment_p.unsqueeze(1)
        # print(f'the shape of segment_p is {segment_p.shape}')
        rendered_image, radii, rendered_depth, rendered_alpha, rendered_segment = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            segment_p=segment_p,
        )

        # print(f'the max of segment_p is {torch.max(segment_p)}')
        # print(f'the min of segment_p is {torch.min(segment_p)}')
        
        # save rendered_segment as gray image
        # print(f'the shape of rendered_segment is {rendered_segment.shape}') # [2, 256, 256]
        # rendered_segment = rendered_segment.cpu().detach().numpy()
        # rendered_segment = rendered_segment[0] * 255
        # rendered_segment = rendered_segment.astype(np.uint8)
        # rendered_segment = Image.fromarray(rendered_segment)
        # rendered_segment.save('rendered_segment.png')
        # raise ValueError('stop here')

        # 将渲染图像的值裁剪到 [0, 1] 范围内，确保图像有效
        
        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "segment": rendered_segment,
        }
