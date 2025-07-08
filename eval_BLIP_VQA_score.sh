#!/bin/bash

# 1. 路径设置
ROOT="./outputs/hcog-generation"
SAMPLE_DIR="./T2I-CompBench/examples/samples"

# 2. 清空样本文件夹
rm -rf "$SAMPLE_DIR"
mkdir -p "$SAMPLE_DIR"

# 3. 开始采样
IMG_IDX=0

find "$ROOT" -mindepth 2 -maxdepth 2 -type d -name "save" | while read SAVE_PARENT; do
    # 找到唯一图片文件夹
    SUB_SAVE=$(find "$SAVE_PARENT" -mindepth 1 -maxdepth 1 -type d | head -n1)
    if [ -z "$SUB_SAVE" ]; then
        echo "没有找到$SAVE_PARENT下的子文件夹，跳过"
        continue
    fi

    # 找图片，随机20张
    IMAGES=($(find "$SUB_SAVE" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort -R | head -20))
    if [ "${#IMAGES[@]}" -eq 0 ]; then
        echo "$SUB_SAVE下无图片，跳过"
        continue
    fi

    # 获取prompt
    PARSED_YAML=$(dirname "$SAVE_PARENT")/configs/parsed.yaml
    if [ ! -f "$PARSED_YAML" ]; then
        echo "$PARSED_YAML 不存在，跳过"
        continue
    fi
    PROMPT=$(yq '.system.prompt_processor.prompt' "$PARSED_YAML")
    [ -z "$PROMPT" ] && PROMPT=$(yq '.prompt' "$PARSED_YAML")
    [ -z "$PROMPT" ] && PROMPT="No prompt found"
    SAFE_PROMPT=$(echo "$PROMPT" | sed 's/\.\s*$//' | tr -d '\n')

    # 拷贝图片
    for IMG in "${IMAGES[@]}"; do
        FILENAME=$(printf "%06d.png" $IMG_IDX)
        OUTNAME="${SAFE_PROMPT}_${FILENAME}"
        cp "$IMG" "$SAMPLE_DIR/$OUTNAME"
        let IMG_IDX++
    done
done

BLIP_LOG="./eval_outputs/blip_vqa_eval.log"
mkdir -p "./eval_outputs"

export project_dir="./T2I-CompBench/BLIPvqa_eval/"
cd "$project_dir" || exit 1
out_dir="../examples/"
python BLIP_vqa.py --out_dir="$out_dir" | tee "$BLIP_LOG"
