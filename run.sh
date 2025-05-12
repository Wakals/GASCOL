#!/bin/bash

# 设定需要监控的 GPU ID （默认 0）
GPU_ID=0

# 设定显存占用阈值（单位 MB，这里是 20000MB = 20GB）
THRESHOLD=22000

# 监测周期（单位秒），可以根据需求修改
INTERVAL=10

# text_prompts=(
#     "an orange cat wearing a yellow suit"
#     "an orange cat wearing a green shirt"
#     "an orange cat wearing a red fireman uniform"
#     "a red rose in a pink vase"
#     "a blue peony in a pink vase"
#     "a yellow tulip in a pink vase"
#     "a pair of red sneakers on a blue chair"
#     "a stack of green books on a blue chair"
#     "a purple gift box on a blue chair"
#     "a red cake in a yellow tray"
#     "a blue spoon in a yellow tray"
#     "a pair of green chopsticks in a yellow tray"
#     "a green vase on a red desk"
#     "a pair of blue sneakers on a red desk"
#     "a yellow tray on a red desk"
#     "a round gift box on a hexagonal table"
#     "a triangular cake on a hexagonal table"
#     "a square tray on a hexagonal table"
#     "a hexagonal cup on a round cabinet"
#     "a triangular sandwich on a round cabinet"
#     "a square bowl on a round cabinet"
#     "a hexagonal cup on a star-shaped tray"
#     "a triangular cake on a star-shaped tray"
#     "a square pepper on a star-shaped tray"
#     "a model of a round house with a hexagonal roof"
#     "a model of a round house with a square roof"
#     "a model of a round house with a spherical roof"
#     "a lego man riding a golden motorcycle"
#     "a silver bunny riding a golden motorcycle"
#     "a wooden dog riding a golden motorcycle"
#     "an orange cat wearing a yellow suit and green sneakers"
#     "an orange cat wearing a yellow suit and red pumps"
#     "an orange cat wearing a yellow suit and cyan boots"
#     "an orange cat wearing a yellow suit and green sneakers and cyan top hat"
#     "an orange cat wearing a yellow suit and green sneakers and pink cap"
#     "an orange cat wearing a yellow suit and green sneakers and red chef's hat"
#     "a blue peony and a yellow tulip in a pink vase"
#     "a red rose and a yellow tulip in a pink vase"
# )

text_prompts=(
    "a wooden dog driving an origami sport car"
    "a metal monkey wearing a golden crown and driving an origami sport car"
    "a metal monkey wearing a chef's hat and driving an origami sport car"
    "a metal monkey wearing a wooden top hat and driving an origami sport car"
    "a lego man driving an origami sport car"
    "a metal monkey driving an origami sport car"
)

while true
do
    # 使用 nvidia-smi 获取指定 GPU 的显存占用（单位 MB）
    # --query-gpu=memory.used 只显示所需的显存使用字段
    # --format=csv,noheader,nounits 去除多余信息，仅显示数值
    usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)

    if [ "$usage" -lt "$THRESHOLD" ]; then
        echo "当前 GPU 显存占用为 ${usage}MB，低于 22GB，开始运行 Python 脚本 ..."
        # CUDA_VISIBLE_DEVICES=4 python launch.py --config custom/threestudio-gaussiandreamer/configs/gaussiandreamer.yaml  --train --gpu 0 system.prompt_processor.prompt="a cartoon girl is smiling, wearing grey shirt, blue skirt, and yellow shoes" system.geometry.geometry_convert_from="shap-e:a cartoon girl wears shirt, skirt, and jacket"
        for prompt in "${text_prompts[@]}"; do
            python launch.py --config custom/threestudio-hcog/configs/hcog.yaml  --train --gpu 0 system.prompt_processor.prompt="$prompt" system.geometry.geometry_convert_from="shap-e:a sport car"
        # 如果只需在检测到一次条件满足后即可退出脚本，则使用 break
        done
        break
    else
        echo "当前 GPU 显存占用为 ${usage}MB，仍然高于 22GB，等待 ${INTERVAL} 秒后再次检测 ..."
        sleep $INTERVAL
    fi
done