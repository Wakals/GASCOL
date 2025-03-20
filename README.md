# HCoG: Apply Hierarchical-Chain-of-Generation to Complex Attributes Text-to-3D Generation

The code is tested on `torch 2.4.1` and `CUDA 12.4`

## Install the requirements

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja
pip install openai==0.28.0
pip install -r requirements_new.txt

cd custom
git clone https://github.com/Wakals/HCoG_SD3.git

git clone --recursive https://github.com/Wakals/GASCOL-diff-gaussian-rasterization.git
pip install ./GASCOL-diff-gaussian-rasterization

pip install ./simple-knn

pip install open3d
pip install pymeshlab
```

## Run the example

```
python launch.py --config custom/threestudio-gaussiandreamer/configs/gaussiandreamer.yaml  --train --gpu 0 system.prompt_processor.prompt="a man in black coat, yellow shirt inside, green hat, blue shoes, and pink trousers is waving" system.geometry.geometry_convert_from="shap-e:a man in shirt, trousers and shoes is waving"
```