#!/bin/bash

# dependency: yq (https://github.com/mikefarah/yq), clip_score

ROOT="./outputs/hcog-generation"
EVAL_ROOT="./eval_outputs"
IMG_DIR="$EVAL_ROOT/clip_score_img"
TXT_DIR="$EVAL_ROOT/clip_score_txt"
LOG_FILE="$EVAL_ROOT/clip_score_log.log"

rm -rf "$EVAL_ROOT"

mkdir -p "$EVAL_ROOT"
mkdir -p "$IMG_DIR" "$TXT_DIR"

if [ "$(ls -A $IMG_DIR)" ]; then
    LAST_IDX=$(ls $IMG_DIR | grep -E '^[0-9]{6}\.png$' | sed 's/\.png$//' | sort | tail -n1)
    START_IDX=$((10#$LAST_IDX + 1))
else
    START_IDX=0
fi

find "$ROOT" -mindepth 2 -maxdepth 2 -type d -name "save" | while read SAVE_PARENT; do
    SUB_SAVE=$(find "$SAVE_PARENT" -mindepth 1 -maxdepth 1 -type d | head -n1)
    if [ -z "$SUB_SAVE" ]; then
        echo "没有找到$SAVE_PARENT下的子文件夹，跳过"
        continue
    fi

    IMAGES=($(find "$SUB_SAVE" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort | head -20))
    if [ "${#IMAGES[@]}" -eq 0 ]; then
        echo "$SUB_SAVE下无图片，跳过"
        continue
    fi

    PARSED_YAML=$(dirname "$SAVE_PARENT")/configs/parsed.yaml
    if [ ! -f "$PARSED_YAML" ]; then
        echo "$PARSED_YAML 不存在，跳过"
        continue
    fi
    PROMPT=$(yq '.system.prompt_processor.prompt' "$PARSED_YAML")
    echo "DEBUG: Extracted prompt is [$PROMPT] from [$PARSED_YAML]"
    [ -z "$PROMPT" ] && PROMPT=$(yq '.prompt' "$PARSED_YAML" 2>/dev/null)
    [ -z "$PROMPT" ] && PROMPT="No prompt found"

    for IMG in "${IMAGES[@]}"; do
        FILENAME=$(printf "%06d.png" $START_IDX)
        cp "$IMG" "$IMG_DIR/$FILENAME"
        echo "$PROMPT" > "$TXT_DIR/${FILENAME%.png}.txt"
        let START_IDX++
    done
done

> "$LOG_FILE"
python -m clip_score "$IMG_DIR" "$TXT_DIR" >> "$LOG_FILE" 2>&1
