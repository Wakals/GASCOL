import os
import cv2
import time
import tqdm
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

import matplotlib.pyplot as plt
from PIL import Image
import glob
import re

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
            self.prompt = self.opt.prompt
            for i in range(len(self.prompt)):
                self.renderer.gaussians.child[i].prompt = self.prompt[i]
                self.renderer.gaussians.child[i].ori = self.opt.ori[i]
                
        if self.opt.floor is not None:
            self.prompt_floor = self.opt.floor

        # override if provide a checkpoint
        if self.opt.load_object is not None:
            for i in range(len(self.opt.load_object)):
                self.renderer.gaussians.child[i].load_ply(self.opt.load_object[i])
            self.renderer.gaussians.floor.load_ply(self.opt.load_floor)            
        else:
            # initialize gaussians
            self.renderer.initialize(num_pts=self.opt.num_pts)

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

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        for chi in self.renderer.gaussians.child:
            chi.training_setup(self.opt)
            chi.active_sh_degree = 0
        if self.opt.floor:
            self.renderer.gaussians.floor.training_setup(self.opt)
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

    def train_step(self):
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
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)

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
                    # print(f'the point is {point}')
                    # print(f'the shape of point is {point.shape}')
                    # print(f'the shape of xyz is {xyz.shape}')
                    # print(f'the shape of scale is {scale.shape}')
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
                SDSloss, _ = self.guidance_cldm.train_step(image, control, step_ratio=step_ratio)
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
    def save_snapshot(self, idx, name=''):
        vers = [-30] * 36
        hors = [i * 10 for i in range(-18, 18)]
        render_resolution = 512
        if not os.path.exists("./image_results/" + self.opt.save_path + name):
            os.makedirs("./image_results/" + self.opt.save_path + name, exist_ok=True)
        if not os.path.exists("./image_results/" + self.opt.save_path + name + "/" + str(idx)):
            os.makedirs("./image_results/" + self.opt.save_path + name + "/" + str(idx), exist_ok=True)
        for i in range(36):
            pose = orbit_camera(vers[i], hors[i], self.opt.radius)

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
            cv2.imwrite(f'./image_results/{self.opt.save_path + name}/{str(idx)}/' + f'{(i - 18) * 10}.png', input_tensor)

    @torch.no_grad()
    def save_model(self):
        os.makedirs('./3d_models/' + self.opt.outdir, exist_ok=True)
        from collections import Counter
        record = Counter()
        for i, chi in enumerate(self.renderer.gaussians.child):
            name = self.prompt[i].replace(" ", "_")
            record[name] += 1
            name += '_' + str(record[name])
            path = './3d_models/' + os.path.join(self.opt.outdir, self.opt.save_path + '_' + name + '_model.ply')
            chi.save_ply(path)
        
        if self.opt.floor:
            self.renderer.gaussians.floor.save_ply('./3d_models/' + os.path.join(self.opt.outdir, self.opt.save_path + '_floor_model.ply'))
        
        # save the whole scene
        path = './3d_models/' + os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
        self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")
    
    # no gui mode
    def train(self, iters=12000):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()

                if (i + 1) % 1000 == 0:
                    self.save_snapshot(i + 1)

            torch.cuda.empty_cache()
            if self.opt.floor:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child] + [self.renderer.gaussians.floor._xyz], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_dc], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._features_rest], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi._scaling for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._scaling], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._rotation], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child]+ [self.renderer.gaussians.floor._opacity], dim=0)
            else:
                self.renderer.gaussians._xyz = torch.cat([chi.get_scene_xyz for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_dc = torch.cat([chi._features_dc for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._features_rest = torch.cat([chi._features_rest for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._scaling = torch.cat([chi.get_scene_scaling for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._rotation = torch.cat([chi.get_scene_rotation for chi in self.renderer.gaussians.child], dim=0)
                self.renderer.gaussians._opacity = torch.cat([chi._opacity for chi in self.renderer.gaussians.child], dim=0)
            
            # save
            self.save_snapshot(0)
            self.save_model()

            for chi in self.renderer.gaussians.child:
                print(f'{chi.prompt} center: {chi.object_center}, scale_factor: {chi.scale_factor}')

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)
    gui.train(opt.iters)
    