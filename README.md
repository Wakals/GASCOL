# HCoG: Apply Hierarchical-Chain-of-Generation to Complex Attributes Text-to-3D Generation

This is the version of HCoG method + GALA3D. The code is tested on `RTX3090` with `Python 3.8`, `torch 2.0.0` and `CUDA 11.8`.

Another version of HCoG + SD3 is in this [repo](https://github.com/Wakals/GASCOL/tree/HCoG_SD3), which is tested on `A100` with `Python 3.11`, `torch 2.4.1` and `CUDA 12.4`.

## Install

```bash
# use new requirements
pip install -r requirements_new.txt

# or use the original cmd
pip install -r requirements.txt
pip install openai==0.28.0

# adan
python3 -m pip install git+https://github.com/sail-sg/Adan.git

# a modified gaussian splatting
git clone --recursive https://github.com/Wakals/GASCOL-diff-gaussian-rasterization.git
mv GASCOL-diff-gaussian-rasterization diff-gaussian-rasterization 
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# MVDream:
git clone https://github.com/bytedance/MVDream.git
pip install ./MVDream

# ControlNet:
git clone https://github.com/lllyasviel/ControlNet-v1-1-nightly.git
pip install ./ControlNet-v1-1-nightly

# lang-sam
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

However, Lang SAM of 0.1.0 may have been deleted, so you can download the source code from [Google Drive](https://drive.google.com/file/d/1Gtql03SyhhNQbVuW3BcMuoY0qgVWjahj/view?usp=sharing), and follow the [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to install it.

To install lang-sam-0.1.0 using the source code, you can first create an Anaconda env, and follow:
```
tar -xzvf lang_sam_0.1.0.tar.gz -C /path/to/new/python/env/lib/python3.8/site-packages/
mkdir /home/user/anaconda3/envs/my_env/lib/python3.8/site-packages/lang_sam-0.1.0.dist-info
echo "Name: lang-sam" > /home/user/anaconda3/envs/my_env/lib/python3.8/site-packages/lang_sam-0.1.0.dist-info/METADATA
```
Following this way, the right structure is:
```
site-packages/
  ├── lang_sam/
  │   ├── __init__.py
  │   ├── some_module.py
  │   └── ...
  ├── lang_sam-0.1.0.dist-info/
  │   ├── METADATA
  │   ├── RECORD
  │   ├── ...
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
- [GALA3D](https://github.com/VDIGPKU/GALA3D)
