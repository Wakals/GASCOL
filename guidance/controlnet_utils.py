from share import *
import cv2
import einops
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.mlsd import MLSDdetector
from diffusers import DDIMScheduler
from annotator.zoe import ZoeDetector

class CLDM(nn.Module):
    def __init__(
            self,
            device,
            t_range=[0.02, 0.98]
    ):
        super().__init__()
        self.device = device

        self.preprocessor = ZoeDetector()

        model_name = 'control_v11f1p_sd15_depth'
        self.model = create_model(f'./ControlNet-v1-1-nightly/models/{model_name}.yaml').cpu()

        self.model.load_state_dict(load_state_dict('./ControlNet-v1-1-nightly/models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        self.model.load_state_dict(load_state_dict(f'./ControlNet-v1-1-nightly/models/{model_name}.pth', location='cuda'), strict=False)

        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.prompt = None
        self.scheduler = DDIMScheduler.from_pretrained(
            'stable-diffusion-2-1-base', subfolder="scheduler", torch_dtype=torch.float32
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

    def refine(self, pred_rgb, layout_image, detect_resolution=512, image_resolution=512, guidance_scale=100, 
             a_prompt='best quality, extremely detailed',
             n_prompt='unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation.',
             num_samples=1, steps=50, guess_mode=False, strength=0.8, eta=0.0, value_threshold=0.1, distance_threshold=0.1):
        prompt = self.prompt
        layout_image = HWC3(layout_image)
        detected_map = self.preprocessor(resize_image(layout_image, detect_resolution))
        # image = Image.fromarray(detected_map)
        # image.save('detected_map.png')
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        batch_size = pred_rgb.shape[0]
        latents = self.encode_imgs(pred_rgb)
        with torch.no_grad():
            self.scheduler.set_timesteps(steps)
            init_step = int(steps * strength)
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

            for i, t in enumerate(self.scheduler.timesteps[init_step:]):
                t = torch.full((num_samples,), t, dtype=torch.long, device=self.device)
                noise_pred_pos = self.model.apply_model(latents, t, cond)
                noise_pred_uncond = self.model.apply_model(latents, t, un_cond)

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                images = self.decode_latents(latents) # [1, 3, 512, 512]
        
        loss = F.mse_loss(pred_rgb, images, reduction='sum')
        images = images.permute(0, 3, 1, 2)
        return loss, images

    # input_image: [1, 3, 512, 512] in [0, 1] cuda tensor
    # layout_image: read PNG file in [0, 255] np.uint8
    def train_step(self, input_image, layout_image, detect_resolution=512, image_resolution=512, guidance_scale=100, 
             a_prompt='best quality, extremely detailed',
             n_prompt='unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation.',
             num_samples=1, ddim_steps=50, guess_mode=False, strength=0.8, eta=0.0, step_ratio=None, value_threshold=0.1, distance_threshold=0.1):
        
        prompt = self.prompt
        layout_image = HWC3(layout_image)
        # detected_map = self.apply_uniformer(resize_image(layout_image, detect_resolution))
        # detected_map = layout_image
        # detected_map = self.apply_mlsd(resize_image(layout_image, detect_resolution), value_threshold, distance_threshold)
        detected_map = self.preprocessor(resize_image(layout_image, detect_resolution))
        # image = Image.fromarray(detected_map)
        # image.save('detected_map.png')
        detected_map = HWC3(detected_map)

        img = resize_image(layout_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        self.ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        if step_ratio is not None:
            # dreamtime-like
            ts = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            ts = torch.full((num_samples,), ts, dtype=torch.long, device=self.device)
        else:
            ts = torch.randint(self.min_step, self.max_step + 1, (num_samples,), dtype=torch.long, device=self.device)

        # to be fed into vae.
        if input_image.shape[2] != 512:
            input_image = F.interpolate(input_image, size=(512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(input_image)

        # predict the noise residual with unet, NO grad
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, ts, noise)
            
            # pred noise
            noise_pred_pos = self.model.apply_model(latents_noisy, ts, cond)
            noise_pred_uncond = self.model.apply_model(latents_noisy, ts, un_cond)

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

            images = self.create_image(latents_noisy, ts, noise_pred).permute(0, 3, 1, 2) # [1, 256, 256, 3]

        grad = (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss, images

    def train_step_image(self, input_image, layout_image, detect_resolution=512, image_resolution=512, guidance_scale=100, 
             a_prompt='best quality, extremely detailed',
             n_prompt='unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, low-resolution, oversaturation.',
             num_samples=1, ddim_steps=50, guess_mode=False, strength=0.8, eta=0.0, step_ratio=None, value_threshold=0.1, distance_threshold=0.1):
        
        prompt = self.prompt
        layout_image = HWC3(layout_image)
        detected_map = self.preprocessor(resize_image(layout_image, detect_resolution))
        # image = Image.fromarray(detected_map)
        # image.save('detected_map.png')
        detected_map = HWC3(detected_map)

        img = resize_image(layout_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        self.ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        if step_ratio is not None:
            # dreamtime-like
            ts = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            ts = torch.full((num_samples,), ts, dtype=torch.long, device=self.device)
        else:
            ts = torch.randint(self.min_step, self.max_step + 1, (num_samples,), dtype=torch.long, device=self.device)

        gamma_t = self.ddim_sampler.sqrt_one_minus_alphas_cumprod[ts] / self.ddim_sampler.sqrt_alphas_cumprod[ts]

        # to be fed into vae.
        if input_image.shape[2] != 512:
            input_image = F.interpolate(input_image, size=(512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(input_image)
        
        # predict the noise residual with unet, NO grad
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, ts, noise)
            
            # pred noise
            noise_pred_pos = self.model.apply_model(latents_noisy, ts, cond)
            noise_pred_uncond = self.model.apply_model(latents_noisy, ts, un_cond)

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

            images = self.create_image(latents_noisy, ts, noise_pred)

        loss = F.mse_loss(input_image, images, reduction='sum') / gamma_t[0]
        # loss = F.mse_loss(input_image, images, reduction='sum')

        images = images.permute(0, 3, 1, 2)
        return loss, images

    def create_image(self, latents, t, noise_pred):

        latents = self.model.predict_start_from_noise(latents, t, noise_pred)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 256, 256]

        return imgs

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 512, 512]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 64, 64]

if __name__ == '__main__':
    device = torch.device('cuda')
    cldm = CLDM(device)
    cldm.eval()
    input_image = torch.rand((1, 3, 512, 512)).cuda()
    layout_image = Image.open('/gala3d/guidance/90.png')
    layout_image = np.array(layout_image).astype(np.uint8)
    loss, img = cldm.sds(input_image, layout_image)
    tensor = img.squeeze().permute(2, 0, 1)
    tensor = (tensor * 255).byte()
    numpy_array = tensor.cpu().numpy()
    image = Image.fromarray(numpy_array)
    print(loss)
    image.save('output.png')
