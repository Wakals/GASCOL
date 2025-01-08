import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import cv2
import time
import tqdm
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

import matplotlib.pyplot as plt
from PIL import Image
import glob
import re

from lang_sam import LangSAM

# SAM_model = LangSAM()

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])  
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size

def map_values(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_point_in_cube(edge):
    randoms = torch.rand(3) - 0.5
    point = randoms * torch.tensor(edge)
    return point.cuda()

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.guidance_sd = None
        self.enable_sd = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree, centers=opt.center, edges=opt.edge, floor=True if self.opt.floor else False)
        self.gaussain_scale_factor = 1

        # input text
        self.prompt = ""
        self.prompt_floor = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # prompt
        if self.opt.prompt is not None:
            print(f'self.opt.prompt is {self.opt.prompt}')
            self.prompt = self.opt.prompt[:]
            for i in range(len(self.prompt)):
                self.renderer.gaussians.child[i].prompt = self.prompt[i]
                self.renderer.gaussians.child[i].ori = self.opt.ori[i]

        if self.opt.edit_prompt is not None:
            self.prompt += self.opt.edit_prompt
                
        if self.opt.floor is not None:
            self.prompt_floor = self.opt.floor

        # override if provide a checkpoint
        if self.opt.load_object is not None:
            for i in range(len(self.opt.load_object)):
                self.renderer.gaussians.child[i].load_ply(self.opt.load_object[i])
            if opt.load_floor:
                self.renderer.gaussians.floor.load_ply(self.opt.load_floor)  
            # color print
            print(f"\033[1;32;40m[INFO] load object from ckpt.\033[0m")
        else:
            # initialize gaussians
            self.renderer.initialize(num_pts=self.opt.num_pts)
            # color print
            print(f"\033[1;32;40m[INFO] initialize objects.\033[0m")

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self, train_stage=1, use_cn=True):

        # self.renderer.gaussians._xyz = torch.empty(0)
        # self.renderer.gaussians._segment_p = torch.empty(0)
        # self.renderer.gaussians._features_dc = torch.empty(0)
        # self.renderer.gaussians._features_rest = torch.empty(0)
        # self.renderer.gaussians._scaling = torch.empty(0)
        # self.renderer.gaussians._rotation = torch.empty(0)
        # self.renderer.gaussians._opacity = torch.empty(0)

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt, stage=train_stage)
        for chi in self.renderer.gaussians.child:
            chi.training_setup(self.opt, stage=train_stage)
            chi.active_sh_degree = 0
        if self.opt.floor:
            self.renderer.gaussians.floor.training_setup(self.opt, stage=train_stage)
            self.renderer.gaussians.floor.active_sh_degree = self.renderer.gaussians.floor.max_sh_degree

        # do not do or do progressive sh-level
        self.renderer.gaussians.active_sh_degree = 0

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""

        print(f"[INFO] loading ControlNet...")
        if use_cn:
            from guidance.controlnet_utils import CLDM
            self.guidance_cldm = CLDM(self.device)
            self.guidance_cldm.prompt = self.opt.scene

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                self.guidance_sd.vae.requires_grad_(False)
                self.guidance_sd.vae.eval()
                print(f"[INFO] loaded SD!")

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds(self.prompt, [self.negative_prompt])
                self.guidance_sd.get_text_embeds(self.prompt_floor, [self.negative_prompt], floor=True)

    def train_step(self, Recur=False):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        relu = torch.nn.ReLU()

        object_SDS_weight = 1
        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            render_resolution = 256
            # avoid too large elevation
            min_ver = -45
            max_ver = 30
            
            # update lr
            for chi in self.renderer.gaussians.child:
                chi.update_learning_rate(self.step)
                if Recur:
                    if len(chi.child) > 0:
                        for chi_child in chi.child:
                            chi_child.update_learning_rate(self.step)
            


            # for chi in self.renderer.gaussians.child:
            #     chi.other_feature_dc.register_hook(print_grad)

            # raise ValueError('stop')

            for i in range(len(self.opt.prompt)):
                loss = 0

                ### novel view (manual batch)
                images = []
                poses = []
                vers, hors, radii = [], [], []

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                # set proper camera
                pose = orbit_camera(self.opt.elevation + ver, hor,
                                    np.linalg.norm(np.array(self.renderer.gaussians.child[i].object_edge)) * 1.5)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=self.renderer.gaussians.child[i])

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i,
                            np.linalg.norm(np.array(self.renderer.gaussians.child[i].object_edge)) * 1.5)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color, gs_model=self.renderer.gaussians.child[i])

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)
                images = torch.cat(images, dim=0)
                poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

                # layout loss
                if self.opt.layout:
                    xyz = self.renderer.gaussians.child[i].get_xyz
                    edge = list(map(lambda x: x * self.opt.layout_scale, self.renderer.gaussians.child[i].object_edge))
                    layoutloss = 0
                    x_up = edge[0] / 2
                    y_up = edge[1] / 2
                    z_up = edge[2] / 2
                    x_low = -edge[0] / 2
                    y_low = -edge[1] / 2
                    z_low = -edge[2] / 2
                    bbox = [x_low, y_low, z_low, x_up, y_up, z_up]
                    for j in range(len(bbox)):
                        sign = 1 if j // 3 == 0 else -1
                        layoutloss += relu((torch.tensor(bbox[j], dtype=torch.float32, device="cuda") - xyz[:, j % 3]) * sign).sum() * 10 ** 3
                        # print(f'layoutloss is {layoutloss}')
                    loss += layoutloss
                
                # guidance loss
                if self.enable_sd:
                    if self.opt.mvdream:
                        SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio, i=i) * object_SDS_weight
                        loss += SDSloss
                    else:
                        SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(image, step_ratio, i=i) * object_SDS_weight
                        loss += SDSloss
                loss.backward()
            
            # raise ValueError('stop')
                                            
            # optimize the whole scene 
            render_resolution = 512

            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)

            step_ratio = min(1, self.step / self.opt.iters)
            loss = 0
            images = []
            images_floor = []
            poses = []
            vers, hors, radii = [], [], []
            hor = np.random.randint(-180, 180)
            ver = np.random.randint(-60, 0)
            radius = 0
            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            if self.opt.regloss:
                for i in range(len(self.opt.prompt)):
                    xyz = self.renderer.gaussians.child[i]._xyz
                    # print(f'the max of x y z is {xyz.max(dim=0)[0]}')
                    # print(f'the min of x y z is {xyz.min(dim=0)[0]}')
                    scale = self.renderer.gaussians.child[i].get_scaling
                    edge = self.renderer.gaussians.child[i].object_edge
                    # print(f'edge is {edge}')
                    normalize_factor = (1 / edge[0] + 1 / edge[1] + 1 / edge[2]) / 3
                    scale = relu(scale - 0.005)
                    # print(scale)
                    # raise ValueError('stop')
                    point = sample_point_in_cube(self.renderer.gaussians.child[i].object_edge)
                    distance_to_point = torch.sum(((point - xyz) * normalize_factor)**2, dim=1, keepdim=True)
                    # print(f'the shape of distance_to_point is {distance_to_point.shape}')
                    # raise ValueError('stop')
                    scale_dis = scale * distance_to_point
                    # print(f'the shape of scale_dis is {scale_dis.shape}')
                    regloss = scale_dis.sum() * 10 ** 3
                    # print(f'regloss is {regloss}')
                    # raise ValueError('stop')
                    loss += regloss
                if self.opt.floor:
                    scale = self.renderer.gaussians.floor.get_scaling
                    # print(f'scale is {scale}')
                    # raise ValueError('stop')
                    regloss = relu(scale - self.opt.floor_scale).sum() * 10 ** 3
                    loss += regloss

            if self.opt.overlap:
                overlap_box_edge = [0.4, 0.2, 0.4]
                obj_xyz = [[[-0.2, -0.3, -0.2], [0.2, -0.1, 0.2]]]
                obj_xyz.append([[-0.2, 0.1, -0.2], [0.2, 0.3, 0.2]])
                obj_xyz = torch.tensor(obj_xyz, dtype=torch.float32, device="cuda")
                center_bias = [torch.tensor([0, -0.2, 0], dtype=torch.float32, device="cuda")]
                center_bias.append(torch.tensor([0, 0.2, 0], dtype=torch.float32, device="cuda"))
                for i in range(len(self.opt.prompt)):
                    xyz = self.renderer.gaussians.child[i]._xyz
                    mask = xyz[:, 0] > obj_xyz[i][0][0]
                    mask = mask & (xyz[:, 0] < obj_xyz[i][1][0])
                    mask = mask & (xyz[:, 1] > obj_xyz[i][0][1])
                    mask = mask & (xyz[:, 1] < obj_xyz[i][1][1])
                    mask = mask & (xyz[:, 2] > obj_xyz[i][0][2])
                    mask = mask & (xyz[:, 2] < obj_xyz[i][1][2])
                    random_point = sample_point_in_cube(overlap_box_edge) + center_bias[i]
                    scale = self.renderer.gaussians.child[i].get_scaling
                    mask = mask.unsqueeze(1).repeat(1, 3)
                    distance_to_point = torch.sum(((random_point - xyz) * mask)**2, dim=1, keepdim=True)
                    # raise ValueError('stop')
                    scale = relu(scale - 0.005)
                    scale_dis = scale * distance_to_point
                    scale_dis = scale_dis * mask
                    overlap_loss = scale_dis.sum() * 100 ** 3
                    loss += overlap_loss
                    # print(f'overlap_loss is {overlap_loss}')

            # sample camera
            pose = orbit_camera(ver, hor, self.opt.radius)

            poses.append(pose)
            cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=self.renderer.gaussians)
            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            images.append(image)

            control = image.squeeze(0).detach().cpu().numpy() * 255.0
            control = control.transpose(1, 2, 0)
            control = control.astype(np.uint8)
            control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(f'image_scene.jpg', control)
            # raise ValueError('stop')

            if self.opt.floor:
                out_floor = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=self.renderer.gaussians.floor)
                image_floor = out_floor["image"].unsqueeze(0)
                images_floor.append(image_floor)
                for view_i in range(1, 4):
                    pose_i = orbit_camera(ver, hor + 90 * view_i, self.opt.radius + radius)
                    poses.append(pose_i)
                    cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    out_i_floor = self.renderer.render(cur_cam_i, bg_color=bg_color, gs_model=self.renderer.gaussians.floor)
                    image_floor = out_i_floor["image"].unsqueeze(0)
                    images_floor.append(image_floor)
                images_floor = torch.cat(images_floor, dim=0)
                poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # SDS loss
            strength = step_ratio * 0.45 + 0.5
            if self.step % self.opt.interval == 0:
                SDSloss, img = self.guidance_cldm.train_step(image, control, step_ratio=step_ratio)
                # Save img
                # tensor = img.squeeze().permute(2, 0, 1)
                # tensor = (tensor * 255).byte()
                # numpy_array = tensor.cpu().numpy()
                # im = Image.fromarray(numpy_array)
                # im.save('output.png')
                # raise ValueError('stop')
                SDSloss *= 0.1 * self.opt.interval
                # SDSloss *= self.opt.interval
                loss += SDSloss
            if self.opt.floor:
                SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(images_floor, poses, step_ratio, i='floor')
                loss += SDSloss
            loss.backward()

            # optimizing
            for chi in self.renderer.gaussians.child:
                chi.optimizer.step()
                chi.optimizer.zero_grad()
            if self.opt.floor:
                self.renderer.gaussians.floor.optimizer.step()
                self.renderer.gaussians.floor.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

    @torch.no_grad()
    def save_snapshot(self, idx, name='', scale=0.9):
        vers = [-30] * 36
        hors = [i * 10 for i in range(-18, 18)]
        render_resolution = 512
        # out_dir = './C_image_results/'
        # out_dir = './M_image_results/'
        # out_dir = './Shape_image_results/'
        out_dir = './CSP_image_results/'
        # out_dir = './random_order_image_results/'
        # out_dir = './image_results/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(out_dir + self.opt.save_path + name):
            os.makedirs(out_dir + self.opt.save_path + name, exist_ok=True)
        if not os.path.exists(out_dir + self.opt.save_path + name + "/" + str(idx)):
            os.makedirs(out_dir + self.opt.save_path + name + "/" + str(idx), exist_ok=True)
        for i in range(36):
            pose = orbit_camera(vers[i], hors[i], self.opt.radius * scale)

            cur_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            
            cur_out = self.renderer.render(cur_cam, gs_model=self.renderer.gaussians)["image"]
            input_tensor = cur_out.detach().cpu().numpy() * 255.0
            input_tensor = input_tensor.transpose(1, 2, 0)
            input_tensor = input_tensor.astype(np.uint8)
            input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
            cv2.imwrite(out_dir + f'{self.opt.save_path + name}/{str(idx)}/' + f'{(i - 18) * 10}.png', input_tensor)

    @torch.no_grad()
    def save_model(self):
        # out_dir = './C_3d_models/'
        # out_dir = './M_3d_models/'
        # out_dir = './Shape_3d_models/'
        out_dir = './CSP_3d_models/'
        # out_dir = './random_order_3d_models/'
        # out_dir = './3d_models/'
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir + self.opt.outdir, exist_ok=True)
        from collections import Counter
        record = Counter()
        for i, chi in enumerate(self.renderer.gaussians.child):
            name = self.prompt[i].replace(" ", "_")
            record[name] += 1
            name += '_' + str(record[name])
            path = out_dir + os.path.join(self.opt.outdir, self.opt.save_path + '_' + name + '_model.ply')
            chi.save_ply(path)
        
        if self.opt.floor:
            self.renderer.gaussians.floor.save_ply(out_dir + os.path.join(self.opt.outdir, self.opt.save_path + '_floor_model.ply'))
        
        # save the whole scene
        path = out_dir + os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
        self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")
    
    # no gui mode
    def train(self, iters=12000):
        if iters > 0:
            self.prepare_train(train_stage=1)
            for i in tqdm.trange(iters):
                self.train_step()

                # print the lr in the optimizer
                for chi in self.renderer.gaussians.child:
                    for param_group in chi.optimizer.param_groups:
                        for param in param_group['params']:
                            if param_group["name"] == "xyz":
                                print(f'Parameter: {param_group["name"]}, Learning Rate: {param_group["lr"]}')

                # raise ValueError('stop')

                if (i + 1) % 1000 == 0 or i == 0:
                    self.save_snapshot(i + 1)

            torch.cuda.empty_cache()
            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)
            
            # save
            self.save_snapshot(0)
            self.save_model()

            for chi in self.renderer.gaussians.child:
                print(f'{chi.prompt} center: {chi.object_center}, scale_factor: {chi.scale_factor}')

    def train_seg_step(self, SAM_model):
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):
            self.seg_step += 1

            render_resolution = 256
            # avoid too large elevation
            min_ver = -30
            max_ver = 45
            
            # update lr
            # for chi in self.renderer.gaussians.child:
            #     chi.update_learning_rate(self.step)

            self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)

            for i in range(1):
                loss = 0.0

                ### novel view (manual batch)
                vers, hors, radii = [], [], []

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                # set proper camera
                # pose = orbit_camera(self.opt.elevation + ver, hor,
                #                     np.linalg.norm(np.array(self.renderer.gaussians.child[i].object_edge)) * 1.5)

                # cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                # bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                # out = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=self.renderer.gaussians.child[i])

                pose = orbit_camera(ver, hor, self.opt.radius)
                
                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=self.renderer.gaussians)

                segment_image = out["segment"] # [2, H, W]
                # use softmax to get the probability
                # print(f'the max of segment_image is {segment_image.max()}')
                # print(f'the min of segment_image is {segment_image.min()}')
                # save segment_image
                # segment_image = softmax(segment_image)
                if self.seg_step % 20 == 0:
                    s_img = segment_image.detach().cpu().numpy()
                    s_img = s_img[0]
                    s_img = s_img * 255
                    s_img = s_img.astype(np.uint8)
                    s_img = Image.fromarray(s_img)
                    s_img.save(f'./segment_image/{self.opt.save_path}_{i}_segment_image.png')
                # raise ValueError('stop')
                image = out["image"] # [3, H, W]

                img = image.detach().cpu().clone().permute(1, 2, 0)
                img = (img * 255).clamp(0, 255).byte()
                np_array = img.numpy()
                image_pil = Image.fromarray(np_array)

                text_prompt = "coat"
                if self.opt.seg_prompt is not None:
                    text_prompt = self.opt.seg_prompt

                masks, _, _, _ = SAM_model.predict(image_pil, text_prompt)
                mask = masks.sum(0)

                if mask.dim() == 0:
                    print('got no mask')
                    continue

                # if masks.shape[0] > 1:
                #     min_sum = 1000000
                #     min_idx = 0
                #     for ii in range(masks.shape[0]):
                #         print(masks[ii].sum())
                #         if masks[ii].sum() < min_sum:
                #             min_sum = masks[ii].sum()
                #             min_idx = ii
                #     mask = masks[min_idx]
                    # mask = torch.zeros_like(masks[0], dtype=torch.bool)
                    # for ii in range(masks.shape[0]):
                    #     if ii is not max_idx:
                    #         mask += masks[ii]
                
                # if masks.shape[0] > 1:
                #     mask = masks[:-1].sum(0)
                
                # connvert segment_image type to float
                mask = mask.float().to(self.device)

                # mask >= 0 and mask <= 1
                mask = mask.clamp(0, 1)
                

                if self.seg_step % 100 == 0:
                    msk = mask.detach().cpu().numpy()
                    msk = msk * 255
                    msk = msk.astype(np.uint8)
                    msk = Image.fromarray(msk)
                    msk.save(f'./mask_image/{self.opt.save_path}_{i}_mask.png')

                seg_loss = F.binary_cross_entropy(segment_image[0], mask)

                seg_loss.backward()

            # optimizing
            for chi in self.renderer.gaussians.child:
                chi.optimizer.step()
                chi.optimizer.zero_grad()
            if self.opt.floor:
                self.renderer.gaussians.floor.optimizer.step()
                self.renderer.gaussians.floor.optimizer.zero_grad()

            # raise ValueError('stop')

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)


    def train_seg(self, iters=200):
        SAM_model = LangSAM()
        self.seg_step = 0
        if iters > 0:
            for chi in self.renderer.gaussians.child:
                chi.reset_segment_p()
            self.renderer.gaussians.reset_segment_p()
            self.prepare_train(train_stage=2)

            for i in tqdm.trange(iters):
                if i >= 300:
                    opt.seg_prompt = 'coat'
                if i >= 600:
                    opt.seg_prompt = 'shoes'
                self.train_seg_step(SAM_model)

            torch.cuda.empty_cache()
            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)

            # save
            self.save_snapshot(0)
            self.save_model()

    def render_part(self):
        idxs = []

        for chi in self.renderer.gaussians.child:
            print(f'the max of segment_p is {chi.get_segment_p.max()}')
            idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.9]
            # remove idx in range(0, 40000)
            # idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.8 and i > 40000]
            print(f'the length of idx is {len(idx)}')
            idxs.append(idx)
            # low_idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] <= 0.97]
            # print(f'the length of low_idx is {len(low_idx)}')
            # mid_idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.6 and chi.get_segment_p[i][0] <= 0.9]
            # print(f'the length of mid_idx is {len(mid_idx)}')

        for chi, idx in zip(self.renderer.gaussians.child, idxs):
            chi.other_xyz = chi._xyz[idx].clone().detach()
            chi.other_segment_p = chi._segment_p[idx].clone().detach()
            chi.other_features_dc = chi._features_dc[idx].clone().detach()
            chi.other_features_rest = chi._features_rest[idx].clone().detach()
            chi.other_scaling = chi._scaling[idx].clone().detach()
            chi.other_rotation = chi._rotation[idx].clone().detach()
            chi.other_opacity = chi._opacity[idx].clone().detach()

        self.renderer.gaussians._xyz = torch.cat([chi.get_scene_other_xyz for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians._segment_p = torch.cat([chi.other_segment_p for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians._features_dc = torch.cat([chi.other_features_dc for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians._features_rest = torch.cat([chi.other_features_rest for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians._scaling = torch.cat([chi.other_scaling for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians._rotation = torch.cat([chi.other_rotation for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians._opacity = torch.cat([chi.other_opacity for chi in self.renderer.gaussians.child], dim=0)
        self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)

        self.save_snapshot(0, name='_part')

    
    def render_clean(self):
        idxs = []

        for chi in self.renderer.gaussians.child:
            print(f'the max of segment_p is {chi.get_segment_p.max()}')
            print(f'the number of points is {chi._segment_p.shape[0]}')
            idx_1 = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.9 and chi.label[i] == 1]
            idx_2 = [i for i in range(chi._segment_p.shape[0]) if chi.label[i] == 0]
            print(f'the number of idx_1 is {len(idx_1)}')
            print(f'the number of idx_2 is {len(idx_2)}')
            idx = idx_1 + idx_2
            # idx = [i for i in range(chi._segment_p.shape[0]) if i < 304230]
            print(f'the length of idx is {len(idx)}')
            # chi.label[idx_1] = 0
            idxs.append(idx)

        for chi, idx in zip(self.renderer.gaussians.child, idxs):
            chi._xyz = chi._xyz[idx].detach()
            print(f'the shape of _xyz is {chi._xyz.shape}')
            chi._segment_p = chi._segment_p[idx].detach()
            chi._features_dc = chi._features_dc[idx].detach()
            chi._features_rest = chi._features_rest[idx].detach()
            chi._scaling = chi._scaling[idx].detach()
            chi._rotation = chi._rotation[idx].detach()
            chi._opacity = chi._opacity[idx].detach()

            chi.label = chi.label[idx].detach()

        if self.opt.floor:
            self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
            self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
            self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
            self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
            self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
            self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
            self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
            self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
        else:
            self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
            print(f'the shape of _xyz is {self.renderer.gaussians._xyz.shape}')
            self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
            self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)

        self.save_snapshot(0, name='_clean')
        # self.save_model()


    def render_seg_scene(self, iters=2500):
        self.prepare_train(train_stage=3)

        idxs = []
        fix_idxs = []

        for chi in self.renderer.gaussians.child:
            idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.75]
            fix_idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] <= 0.75]
            idxs.append(idx)
            fix_idxs.append(fix_idx)
            print(f'the length of idx is {len(idx)}')
            print(f'the length of fix_idx is {len(fix_idx)}')

        for chi, idx, fix_idx in zip(self.renderer.gaussians.child, idxs, fix_idxs):
            chi.other_xyz = chi._xyz[idx].clone().detach().requires_grad_(True)
            chi.other_segment_p = chi._segment_p[idx].clone().detach().requires_grad_(True)
            chi.other_features_dc = chi._features_dc[idx].clone().detach().requires_grad_(True)
            chi.other_features_rest = chi._features_rest[idx].clone().detach().requires_grad_(True)
            chi.other_scaling = chi._scaling[idx].clone().detach().requires_grad_(True)
            chi.other_rotation = chi._rotation[idx].clone().detach().requires_grad_(True)
            chi.other_opacity = chi._opacity[idx].clone().detach().requires_grad_(True)
            chi.fix_xyz = chi._xyz[fix_idx].clone().detach().requires_grad_(False)
            chi.fix_segment_p = chi._segment_p[fix_idx].clone().detach().requires_grad_(False)
            chi.fix_features_dc = chi._features_dc[fix_idx].clone().detach().requires_grad_(False)
            chi.fix_features_rest = chi._features_rest[fix_idx].clone().detach().requires_grad_(False)
            chi.fix_scaling = chi._scaling[fix_idx].clone().detach().requires_grad_(False)
            chi.fix_rotation = chi._rotation[fix_idx].clone().detach().requires_grad_(False)
            chi.fix_opacity = chi._opacity[fix_idx].clone().detach().requires_grad_(False)

        for chi in self.renderer.gaussians.child:
            chi._xyz = torch.cat([chi.other_xyz, chi.fix_xyz], dim=0)
            chi._segment_p = torch.cat([chi.other_segment_p, chi.fix_segment_p], dim=0)
            chi._features_dc = torch.cat([chi.other_features_dc, chi.fix_features_dc], dim=0)
            chi._features_rest = torch.cat([chi.other_features_rest, chi.fix_features_rest], dim=0)
            chi._scaling = torch.cat([chi.other_scaling, chi.fix_scaling], dim=0)
            chi._rotation = torch.cat([chi.other_rotation, chi.fix_rotation], dim=0)
            chi._opacity = torch.cat([chi.other_opacity, chi.fix_opacity], dim=0)

        self.renderer.gaussians.training_setup(self.opt, stage=3)
        for chi in self.renderer.gaussians.child:
            chi.training_setup(self.opt, stage=3)
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()

                for chi in self.renderer.gaussians.child:
                    chi._xyz = torch.cat([chi.other_xyz, chi.fix_xyz], dim=0)
                    chi._segment_p = torch.cat([chi.other_segment_p, chi.fix_segment_p], dim=0)
                    chi._features_dc = torch.cat([chi.other_features_dc, chi.fix_features_dc], dim=0)
                    chi._features_rest = torch.cat([chi.other_features_rest, chi.fix_features_rest], dim=0)
                    chi._scaling = torch.cat([chi.other_scaling, chi.fix_scaling], dim=0)
                    chi._rotation = torch.cat([chi.other_rotation, chi.fix_rotation], dim=0)
                    chi._opacity = torch.cat([chi.other_opacity, chi.fix_opacity], dim=0)

                if (i + 1) % 500 == 0 or i == 0:
                    self.save_snapshot(i + 1)

            torch.cuda.empty_cache()
            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)
            
            # save
            self.save_snapshot(0)
            self.save_model()

            for chi in self.renderer.gaussians.child:
                print(f'{chi.prompt} center: {chi.object_center}, scale_factor: {chi.scale_factor}')

    def train_render_seg_scene(self, iters=3000):

        self.renderer.gaussians.spatial_lr_scale = 0.001
        self.renderer.gaussians.color_lr_scale = 2.0
        for chi in self.renderer.gaussians.child:
            chi.spatial_lr_scale = 0.001
            chi.color_lr_scale = 2.0

        self.prepare_train(train_stage=3, use_cn=False)

        print(f"Loading More ...")
        from guidance.optim_model import CLDM_O
        self.cldm_o = CLDM_O(self.device)
        self.cldm_o.prompt = self.opt.edit_prompt[0]

        assert len(self.renderer.gaussians.child) == 1

        idxs = []
        fix_idxs = []

        for chi in self.renderer.gaussians.child:
            idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.9]
            fix_idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] <= 0.9 and chi.label[i] == 0]
            idxs.append(idx)
            fix_idxs.append(fix_idx)
            print(f'the length of idx is {len(idx)}')
            print(f'the length of fix_idx is {len(fix_idx)}')
            label_idx = [i for i in range(chi._segment_p.shape[0]) if chi.get_segment_p[i][0] > 0.9 and chi.label[i] == 1] + [i for i in range(chi._segment_p.shape[0]) if chi.label[i] == 0]

        for chi, idx, fix_idx in zip(self.renderer.gaussians.child, idxs, fix_idxs):
            chi.other_xyz = chi._xyz[idx].clone().detach().requires_grad_(True)
            chi.other_segment_p = chi._segment_p[idx].clone().detach().requires_grad_(True)
            chi.other_features_dc = chi._features_dc[idx].clone().detach().requires_grad_(True)
            chi.other_features_rest = chi._features_rest[idx].clone().detach().requires_grad_(True)
            chi.other_scaling = chi._scaling[idx].clone().detach().requires_grad_(True)
            chi.other_rotation = chi._rotation[idx].clone().detach().requires_grad_(True)
            chi.other_opacity = chi._opacity[idx].clone().detach().requires_grad_(True)
            chi.fix_xyz = chi._xyz[fix_idx].clone().detach().requires_grad_(True)
            chi.fix_segment_p = chi._segment_p[fix_idx].clone().detach().requires_grad_(True)
            chi.fix_features_dc = chi._features_dc[fix_idx].clone().detach().requires_grad_(True)
            chi.fix_features_rest = chi._features_rest[fix_idx].clone().detach().requires_grad_(True)
            chi.fix_scaling = chi._scaling[fix_idx].clone().detach().requires_grad_(True)
            chi.fix_rotation = chi._rotation[fix_idx].clone().detach().requires_grad_(True)
            chi.fix_opacity = chi._opacity[fix_idx].clone().detach().requires_grad_(True)

            chi.label = chi.label[label_idx].clone().detach()

        self.renderer.create_children()

        for chi in self.renderer.gaussians.child:
            chi._xyz = torch.cat([ch._xyz for ch in chi.child] + [chi.fix_xyz], dim=0)
            chi._segment_p = torch.cat([ch._segment_p for ch in chi.child] + [chi.fix_segment_p], dim=0)
            chi._features_dc = torch.cat([ch._features_dc for ch in chi.child]+ [chi.fix_features_dc], dim=0)
            chi._features_rest = torch.cat([ch._features_rest for ch in chi.child]+ [chi.fix_features_rest], dim=0)
            chi._scaling = torch.cat([ch._scaling for ch in chi.child]+ [chi.fix_scaling], dim=0)
            chi._rotation = torch.cat([ch._rotation for ch in chi.child]+ [chi.fix_rotation], dim=0)
            chi._opacity = torch.cat([ch._opacity for ch in chi.child]+ [chi.fix_opacity], dim=0)

        self.renderer.gaussians.training_setup(self.opt, stage=3)
        for chi in self.renderer.gaussians.child:
            chi.training_setup(self.opt, stage=3)
            if len(chi.child) > 0:
                for ch in chi.child:
                    ch.training_setup(self.opt, stage=1)
        if iters > 0:
            for i in tqdm.trange(iters):
                # densify the gaussians
                # if i > 0 and i <= 4000 and i % 5 == 0:
                if i == 1:
                    for chi in self.renderer.gaussians.child:
                        print(f'before densify, the num of points of chi is {chi.other_xyz.shape[0]}')
                        chi.densify_chi_child()
                        print(f'after densify, the num of points of chi is {chi.other_xyz.shape[0]}')

                self.train_render_seg_scene_step()

                for chi in self.renderer.gaussians.child:
                    chi._xyz = torch.cat([ch._xyz for ch in chi.child] + [chi.fix_xyz], dim=0)
                    chi._segment_p = torch.cat([ch._segment_p for ch in chi.child] + [chi.fix_segment_p], dim=0)
                    chi._features_dc = torch.cat([ch._features_dc for ch in chi.child] + [chi.fix_features_dc], dim=0)
                    chi._features_rest = torch.cat([ch._features_rest for ch in chi.child] + [chi.fix_features_rest], dim=0)
                    chi._scaling = torch.cat([ch._scaling for ch in chi.child] + [chi.fix_scaling], dim=0)
                    chi._rotation = torch.cat([ch._rotation for ch in chi.child] + [chi.fix_rotation], dim=0)
                    chi._opacity = torch.cat([ch._opacity for ch in chi.child] + [chi.fix_opacity], dim=0)

                if (i + 1) % 500 == 0 or i == 0:
                    self.save_snapshot(i + 1)

            torch.cuda.empty_cache()
            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)
            
            # save
            self.save_snapshot(0)
            self.save_model()

            for chi in self.renderer.gaussians.child:
                print(f'{chi.prompt} center: {chi.object_center}, scale_factor: {chi.scale_factor}')

    def train_render_seg_scene_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        relu = torch.nn.ReLU()

        object_SDS_weight = 1
        for _ in range(self.train_steps):
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            render_resolution = 256
            # avoid too large elevation
            min_ver = -45
            max_ver = 30
            
            # update lr
            # print(f'max step is {self.opt.position_lr_max_steps}')
            for chi in self.renderer.gaussians.child:
                chi.update_learning_rate(self.step)
                if len(chi.child) > 0:
                    for chi_child in chi.child:
                        chi_child.update_learning_rate(self.step)

            if self.step % 1 == 0:

                for i in range(len(self.opt.prompt)):
                    loss = 0

                    ### novel view (manual batch)
                    images = []
                    poses = []
                    vers, hors, radii = [], [], []

                    # render random view
                    ver = np.random.randint(min_ver, max_ver)
                    hor = np.random.randint(-180, 180)
                    radius = 0

                    vers.append(ver)
                    hors.append(hor)
                    radii.append(radius)

                    # set proper camera
                    pose = orbit_camera(self.opt.elevation + ver, hor,
                                        np.linalg.norm(np.array(self.renderer.gaussians.child[i].object_edge)) * 1.5)
                    poses.append(pose)

                    cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                    out = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=self.renderer.gaussians.child[i])

                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)

                    # enable mvdream training
                    if self.opt.mvdream:
                        for view_i in range(1, 4):
                            pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i,
                                np.linalg.norm(np.array(self.renderer.gaussians.child[i].object_edge)) * 1.5)
                            poses.append(pose_i)

                            cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                            out_i = self.renderer.render(cur_cam_i, bg_color=bg_color, gs_model=self.renderer.gaussians.child[i])

                            image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                            images.append(image)
                    images = torch.cat(images, dim=0)
                    poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

                    # layout loss
                    if self.opt.layout:
                        xyz = self.renderer.gaussians.child[i].get_xyz
                        edge = list(map(lambda x: x * self.opt.layout_scale, self.renderer.gaussians.child[i].object_edge))
                        layoutloss = 0
                        x_up = edge[0] / 2
                        y_up = edge[1] / 2
                        z_up = edge[2] / 2
                        x_low = -edge[0] / 2
                        y_low = -edge[1] / 2
                        z_low = -edge[2] / 2
                        bbox = [x_low, y_low, z_low, x_up, y_up, z_up]
                        for j in range(len(bbox)):
                            sign = 1 if j // 3 == 0 else -1
                            layoutloss += relu((torch.tensor(bbox[j], dtype=torch.float32, device="cuda") - xyz[:, j % 3]) * sign).sum() * 10 ** 3
                            # print(f'layoutloss is {layoutloss}')
                        loss += layoutloss
                    
                    # guidance loss
                    if self.enable_sd:
                        if self.opt.mvdream:
                            SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio, i=i) * object_SDS_weight
                            loss += SDSloss
                        else:
                            SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(image, step_ratio, i=i) * object_SDS_weight
                            loss += SDSloss
                    loss.backward()


            for chi in self.renderer.gaussians.child:
                if len(chi.child) > 0:
                    for i in range(len(chi.child)):
                        chi_child = chi.child[i]
                        loss = 0

                        ### novel view (manual batch)
                        images = []
                        poses = []
                        vers, hors, radii = [], [], []

                        # render random view
                        ver = np.random.randint(min_ver, max_ver)
                        hor = np.random.randint(-180, 180)
                        radius = 0

                        vers.append(ver)
                        hors.append(hor)
                        radii.append(radius)

                        # set proper camera
                        pose = orbit_camera(self.opt.elevation + ver, hor,
                                            np.linalg.norm(np.array(chi_child.object_edge)) * 1.5)
                        poses.append(pose)

                        cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        out = self.renderer.render(cur_cam, bg_color=bg_color, gs_model=chi_child)

                        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

                        # enable mvdream training
                        if self.step < 100 or self.step % 10 == 0:
                            if self.opt.mvdream:
                                for view_i in range(1, 4):
                                    pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i,
                                        np.linalg.norm(np.array(chi_child.object_edge)) * 1.5)
                                    poses.append(pose_i)

                                    cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                                    out_i = self.renderer.render(cur_cam_i, bg_color=bg_color, gs_model=chi_child)

                                    image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                                    images.append(image)
                            images = torch.cat(images, dim=0)
                            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

                        else:
                            control = image.squeeze(0).detach().cpu().numpy() * 255.0
                            control = control.transpose(1, 2, 0)
                            control = control.astype(np.uint8)
                            control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)

                        # layout loss
                        if self.opt.layout:
                            xyz = chi_child.get_xyz
                            edge = list(map(lambda x: x * self.opt.layout_scale, chi_child.object_edge))
                            layoutloss = 0
                            x_up = edge[0] / 2
                            y_up = edge[1] / 2
                            z_up = edge[2] / 2
                            x_low = -edge[0] / 2
                            y_low = -edge[1] / 2
                            z_low = -edge[2] / 2
                            bbox = [x_low, y_low, z_low, x_up, y_up, z_up]
                            for j in range(len(bbox)):
                                sign = 1 if j // 3 == 0 else -1
                                layoutloss += relu((torch.tensor(bbox[j], dtype=torch.float32, device="cuda") - xyz[:, j % 3]) * sign).sum() * 10 ** 3
                                # print(f'layoutloss is {layoutloss}')
                            loss += layoutloss
                        
                        # guidance loss
                        if self.step < 100 or self.step % 10 == 0:
                            if self.enable_sd:
                                if self.opt.mvdream:
                                    SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio, i=i+len(self.opt.prompt)) * object_SDS_weight
                                    loss += SDSloss
                                else:
                                    SDSloss = self.opt.lambda_sd * self.guidance_sd.train_step(image, step_ratio, i=i+len(self.opt.prompt)) * object_SDS_weight
                                    loss += SDSloss
                        else:
                            SDSloss, _ = self.cldm_o.train_step(image, control, step_ratio=step_ratio)
                            loss += SDSloss
                        loss.backward()


            render_resolution = 512

            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)

            step_ratio = min(1, self.step / self.opt.iters)
            loss = 0
            images = []
            images_floor = []
            poses = []
            vers, hors, radii = [], [], []
            hor = np.random.randint(-180, 180)
            ver = np.random.randint(-60, 0)
            radius = 0
            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            if self.step % 1 == 0:

                if self.opt.regloss:
                    for i in range(len(self.opt.prompt)):
                        if len(self.renderer.gaussians.child[i].child) > 0:
                            for chi_child in self.renderer.gaussians.child[i].child:
                                xyz = chi_child._xyz
                                scale = chi_child.get_scaling
                                edge = chi_child.object_edge
                                normalize_factor = (1 / edge[0] + 1 / edge[1] + 1 / edge[2]) / 3
                                scale = relu(scale - 0.005)
                                point = sample_point_in_cube(chi_child.object_edge)
                                distance_to_point = torch.sum(((point - xyz) * normalize_factor)**2, dim=1, keepdim=True)
                                scale_dis = scale * distance_to_point
                                regloss = scale_dis.sum() * 10 ** 3
                                loss += regloss
                        else:
                            xyz = self.renderer.gaussians.child[i]._xyz
                            scale = self.renderer.gaussians.child[i].get_scaling
                            edge = self.renderer.gaussians.child[i].object_edge
                            normalize_factor = (1 / edge[0] + 1 / edge[1] + 1 / edge[2]) / 3
                            scale = relu(scale - 0.005)
                            point = sample_point_in_cube(self.renderer.gaussians.child[i].object_edge)
                            distance_to_point = torch.sum(((point - xyz) * normalize_factor)**2, dim=1, keepdim=True)
                            scale_dis = scale * distance_to_point
                            regloss = scale_dis.sum() * 10 ** 3
                            loss += regloss
                    if self.opt.floor:
                        scale = self.renderer.gaussians.floor.get_scaling
                        regloss = relu(scale - self.opt.floor_scale).sum() * 10 ** 3
                        loss += regloss
                loss.backward()
                            

            # optimizing
            for chi in self.renderer.gaussians.child:
                # print("SDSSDS--------------------------")
                chi.optimizer.step()
                chi.optimizer.zero_grad()
                if len(chi.child) > 0:
                    for chi_child in chi.child:
                        chi_child.optimizer.step()
                        chi_child.optimizer.zero_grad()
            if self.opt.floor:
                self.renderer.gaussians.floor.optimizer.step()
                self.renderer.gaussians.floor.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True


    def train_extend(self, iters=5000):

        self.renderer.gaussians.spatial_lr_scale = 2.0
        self.renderer.gaussians.color_lr_scale = 1.0
        for chi in self.renderer.gaussians.child:
            chi.spatial_lr_scale = 2.0
            chi.color_lr_scale = 1.0

        if iters > 0:
            self.prepare_train(train_stage=3)

            assert len(self.renderer.gaussians.child) == 1
            for chi in self.renderer.gaussians.child:
                chi.label = torch.zeros((chi.label.shape[0], 1), dtype=torch.bool, device="cuda")

                # densify
                chi.densify_directly()
                
                chi.fix_xyz = chi._xyz.clone().detach().requires_grad_(False)
                chi.fix_segment_p = chi._segment_p.clone().detach().requires_grad_(False)
                chi.fix_features_dc = chi._features_dc.clone().detach().requires_grad_(False)
                chi.fix_features_rest = chi._features_rest.clone().detach().requires_grad_(False)
                chi.fix_scaling = chi._scaling.clone().detach().requires_grad_(False)
                chi.fix_rotation = chi._rotation.clone().detach().requires_grad_(False)
                chi.fix_opacity = chi._opacity.clone().detach().requires_grad_(False)

                chi._xyz = torch.cat([chi.other_xyz, chi.fix_xyz], dim=0)
                chi._segment_p = torch.cat([chi.other_segment_p, chi.fix_segment_p], dim=0)
                chi._features_dc = torch.cat([chi.other_features_dc, chi.fix_features_dc], dim=0)
                chi._features_rest = torch.cat([chi.other_features_rest, chi.fix_features_rest], dim=0)
                chi._scaling = torch.cat([chi.other_scaling, chi.fix_scaling], dim=0)
                chi._rotation = torch.cat([chi.other_rotation, chi.fix_rotation], dim=0)
                chi._opacity = torch.cat([chi.other_opacity, chi.fix_opacity], dim=0)

            for chi in self.renderer.gaussians.child:
                chi._xyz = torch.cat([chi.other_xyz, chi.fix_xyz], dim=0)
                chi._segment_p = torch.cat([chi.other_segment_p, chi.fix_segment_p], dim=0)
                chi._features_dc = torch.cat([chi.other_features_dc, chi.fix_features_dc], dim=0)
                chi._features_rest = torch.cat([chi.other_features_rest, chi.fix_features_rest], dim=0)
                chi._scaling = torch.cat([chi.other_scaling, chi.fix_scaling], dim=0)
                chi._rotation = torch.cat([chi.other_rotation, chi.fix_rotation], dim=0)
                chi._opacity = torch.cat([chi.other_opacity, chi.fix_opacity], dim=0)

            self.renderer.gaussians.training_setup(self.opt, stage=3)
            for chi in self.renderer.gaussians.child:
                chi.training_setup(self.opt, stage=3)
            # before_param = self.renderer.gaussians.child[0]._xyz[40000:].detach().clone()
            for i in tqdm.trange(iters):
                
                self.train_step()
                # if i % 1 == 0:
                #     after_param = self.renderer.gaussians.child[0]._xyz[40000:].detach().clone()
                #     if torch.equal(before_param, after_param):
                #         print(f'_xyz has not changed')
                #     else:
                #         print(f'_xyz has changed')

                for chi in self.renderer.gaussians.child:
                    chi._xyz = torch.cat([chi.other_xyz, chi.fix_xyz], dim=0)
                    chi._segment_p = torch.cat([chi.other_segment_p, chi.fix_segment_p], dim=0)
                    chi._features_dc = torch.cat([chi.other_features_dc, chi.fix_features_dc], dim=0)
                    chi._features_rest = torch.cat([chi.other_features_rest, chi.fix_features_rest], dim=0)
                    chi._scaling = torch.cat([chi.other_scaling, chi.fix_scaling], dim=0)
                    chi._rotation = torch.cat([chi.other_rotation, chi.fix_rotation], dim=0)
                    chi._opacity = torch.cat([chi.other_opacity, chi.fix_opacity], dim=0)

                if (i + 1) % 500 == 0 or i == 0:
                    self.save_snapshot(i + 1)

            torch.cuda.empty_cache()
            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._segment_p], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor.label], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._segment_p = torch.cat([chi._segment_p for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians.label = torch.cat([chi.label for chi in self.renderer.gaussians.child], dim=0)
            
            # save
            self.save_snapshot(0)
            self.save_model()

            for chi in self.renderer.gaussians.child:
                print(f'{chi.prompt} center: {chi.object_center}, scale_factor: {chi.scale_factor}')


def gpt_response():
    """
    use GPT to generate the hierarchical chain of generation
    """
    pass


def pipeline(gui):
    attributes = []
    attributes = gpt_response()
    gui.train(opt.iters)
    for attr in attributes:
        if attr == "extend":
            gui.train_extend(iters=15000)
        else:
            gui.train_seg(iters=200)
            gui.train_render_seg_scene(iters=3000)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)
    """
    use the whole pipeline
    """
    pipeline(gui)

    """
    break down the pipeline
    """
    # gui.train(opt.iters)
    # gui.train_extend(iters=15000)
    # gui.train_seg(iters=200)
    # gui.render_seg_scene()
    # gui.train_render_seg_scene(iters=2500)

    # gui.render_part()
    # gui.render_clean()
    