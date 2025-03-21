# HCoG: Apply Hierarchical-Chain-of-Generation to Complex Attributes Text-to-3D Generation

This is the version of HCoG method + Stable Diffusion v3. The code is tested on `A100`, `Python 3.11`, `torch 2.4.1` and `CUDA 12.4`.

Another version of HCoG + GALA3D is in this [repo](https://github.com/Wakals/GASCOL/tree/main), which is tested on `RTX3090` with `Python 3.8`, `torch 2.0.0` and `CUDA 11.8`.

## Install the requirements

The requirements is heavily based on [Threestudio](https://github.com/threestudio-project/threestudio).
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install ninja
pip install openai==0.28.0
pip install -r requirements.txt
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

cd custom
git clone https://github.com/Wakals/HCoG_SD3.git
mv HCoG_SD3 threestudio-hcog
cd threestudio-hcog

git clone --recursive https://github.com/Wakals/GASCOL-diff-gaussian-rasterization.git
mv GASCOL-diff-gaussian-rasterization diff-gaussian-rasterization      
pip install ./diff-gaussian-rasterization

pip install ./simple-knn

pip install open3d
pip install pymeshlab
```

## Configure OpenAI's API key

Our code is constructed on the version of `openai==0.28.0`, and the code to call the API can be found in [./threestudio/gpt/PE.py](./threestudio/gpt/PE.py). You should get your api key from [OpenAI API Platform](https://platform.openai.com/api-keys), putting it at L12 in [PE.py](./threestudio/gpt/PE.py) and api base website at L63 in [PE.py](./threestudio/gpt/PE.py).

If you have difficulty of getting api key. You can check the example in [PE.py](./threestudio/gpt/PE.py) and use your convenient large model to get a generation order and fill it in according to the format.

## Run the example

For recreating the example, run:
```
python launch.py --config custom/threestudio-hcog/configs/hcog.yaml  --train --gpu 0 system.prompt_processor.prompt="a man in black coat, yellow shirt inside, green hat, blue shoes, and pink trousers is waving" system.geometry.geometry_convert_from="shap-e:a man in shirt, trousers and shoes is waving"
```