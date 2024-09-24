# GALA3D

This repository contains the official implementation for [ICML 2024 | GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting](https://arxiv.org/abs/2402.07207).

### [Project Page](https://gala3d.github.io/) | [Arxiv](https://arxiv.org/abs/2402.07207) | [Code](https://github.com/VDIGPKU/GALA3D)

## Cloning the Repo

This repository contains the implementation associated with the paper "GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting". Please be advised that the use of this code is strictly limited to non-commercial purposes.

In compliance with licensing agreements and open-source constraints, certain codes and modules within this repo have been modified or removed. These could lead to issues. If you have any further questions, please feel free to send your questions to [xy56.zhou@gmail.com]. The complete code will be released in our future work.

We welcome all kinds of exchanges and discussions that will help us improve and refine this project. Your insights and feedback are invaluable as we strive for excellence and clarity in our work.

## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

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

- Ubuntu 22 with torch 2.2.2 & CUDA 12.1 on A800.

## Prepare pretrained models

- MVDream can be downloaded [here](https://huggingface.co/MVDream/MVDream/tree/main).

- SD weights required by ControlNet can be downloaded [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

- ControlNet can be downloaded [here](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main).

```
GALA3D
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

Layout interpreted by LLMs

Here is an easy example of the coarse layout interpretation using LLM:
```bash
"Generate a 3D bedroom for me, which contains a bed, two wooden nightstands, a square wooden table with a table lamp on it, and a wooden wardrobe. Please extract the instance objects from the provided description and generate the Blender code describing the layout size and position of each object in the scene. The layout is represented by cuboid bounding boxes. The 3D scene contains the following key parameters in the following format: object_class: [text describing the object class], layout_box_center: [0.0, 0.0, 0.0], layout_box_scale [0.0, 0.0, 0.0], where object_class corresponds to the extracted object class and quantity. Each layout should not be nested within another and should adhere to typical real-world distribution and proportions. The dimensions of all layouts should be within [-1, 1]."
```

You can use any LLM model to generate layouts and test their quality of generation. Note that you need to standardize the output layout parameters to the format specified in the configs, which may require some adjustments to the prompts and formatting conversions.

3D Scene Generation:
```bash
### training
python main.py --config configs/bedroom.yaml
```

- Once the training is completed, you can find the image results in the image_results directory and the 3D Gaussians in the 3d_models directory.

Please check `./configs/bedroom.yaml` for more options.

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [dreamgaussian](https://github.com/dreamgaussian/dreamgaussian)

## Citation

```
@article{zhou2024gala3d,
  title={GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting},
  author={Zhou, Xiaoyu and Ran, Xingjian and Xiong, Yajiao and He, Jinlin and Lin, Zhiwei and Wang, Yongtao and Sun, Deqing and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2402.07207},
  year={2024}
}
```
