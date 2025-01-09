# GASCOL

## Install

```bash
# use new requirements
pip install -r requirements_new.txt

# or use the original cmd
pip install -r requirements.txt
pip install openai==0.28.0

# a modified gaussian splatting
git clone --recursive https://github.com/Wakals/GASCOL-diff-gaussian-rasterization.git
pip install ./GASCOL-diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# MVDream:
git clone https://github.com/bytedance/MVDream.git
pip install ./MVDream

# ControlNet:
git clone https://github.com/lllyasviel/ControlNet-v1-1-nightly.git
pip install ./ControlNet-v1-1-nightly
```

Tested on:

- Ubuntu 22 with torch 1.12.1 & CUDA 11.6.

## Prepare pretrained models

- MVDream can be downloaded [here](https://huggingface.co/MVDream/MVDream/tree/main).

- SD weights required by ControlNet can be downloaded [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

- ControlNet can be downloaded [here](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main).

```
GASCOL
|---MVDream
|   |---sd-v2.1-base-4view.pth
|   |---...
|---ControlNet-v1-1-nightly
    |---models
        |---control_v11f1p_sd15_depth.yaml
        |---control_v11f1p_sd15_depth.pth
        |---v1-5-pruned.ckpt
        |---... (if you want some other controls)
```

## Usage

3D Scene Generation:
```bash
### training
python main.py --config configs/man.yaml
```

- Once the training is completed, you can find the image results in the image_results directory and the 3D Gaussians in the 3d_models directory.

Please check `./configs/man.yaml` for more options.

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [dreamgaussian](https://github.com/dreamgaussian/dreamgaussian)
