#!/bin/bash

# 设定需要监控的 GPU ID （默认 0）
GPU_ID=2

# 设定显存占用阈值（单位 MB，这里是 20000MB = 20GB）
THRESHOLD=22000

# 监测周期（单位秒），可以根据需求修改
INTERVAL=10

while true
do
    # 使用 nvidia-smi 获取指定 GPU 的显存占用（单位 MB）
    # --query-gpu=memory.used 只显示所需的显存使用字段
    # --format=csv,noheader,nounits 去除多余信息，仅显示数值
    usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)

    if [ "$usage" -lt "$THRESHOLD" ]; then
        echo "当前 GPU 显存占用为 ${usage}MB，低于 20GB，开始运行 Python 脚本 ..."
        # CUDA_VISIBLE_DEVICES=4 python launch.py --config custom/threestudio-gaussiandreamer/configs/gaussiandreamer.yaml  --train --gpu 0 system.prompt_processor.prompt="a cartoon girl is smiling, wearing grey shirt, blue skirt, and yellow shoes" system.geometry.geometry_convert_from="shap-e:a cartoon girl wears shirt, skirt, and jacket"
        CUDA_VISIBLE_DEVICES=2 python launch.py --config custom/threestudio-gaussiandreamer/configs/gaussiandreamer.yaml  --train --gpu 0 system.prompt_processor.prompt="a man in black coat, yellow shirt inside, green hat, blue shoes, and pink trousers is waving" system.geometry.geometry_convert_from="shap-e:a man in shirt, trousers and shoes is waving"
        # 如果只需在检测到一次条件满足后即可退出脚本，则使用 break
        break
    else
        echo "当前 GPU 显存占用为 ${usage}MB，仍然高于 20GB，等待 ${INTERVAL} 秒后再次检测 ..."
        sleep $INTERVAL
    fi
done