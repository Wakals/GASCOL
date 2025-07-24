#!/bin/bash

ROOT="./outputs/hcog-generation"
SAMPLE_DIR="./T2I-CompBench/examples/samples"

rm -rf "$SAMPLE_DIR"
mkdir -p "$SAMPLE_DIR"

IMG_IDX=0

find "$ROOT" -mindepth 2 -maxdepth 2 -type d -name "save" | while read SAVE_PARENT; do
    SUB_SAVE=$(find "$SAVE_PARENT" -mindepth 1 -maxdepth 1 -type d | head -n1)
    if [ -z "$SUB_SAVE" ]; then
        echo "not found sub folder under $SAVE_PARENT, skip"
        continue
    fi

    IMAGES=($(find "$SUB_SAVE" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort -R | head -20))
    if [ "${#IMAGES[@]}" -eq 0 ]; then
        echo "no images under $SUB_SAVE, skip"
        continue
    fi

    PARSED_YAML=$(dirname "$SAVE_PARENT")/configs/parsed.yaml
    if [ ! -f "$PARSED_YAML" ]; then
        echo "$PARSED_YAML not found, skip"
        continue
    fi
    PROMPT=$(yq '.system.prompt_processor.prompt' "$PARSED_YAML")
    [ -z "$PROMPT" ] && PROMPT=$(yq '.prompt' "$PARSED_YAML")
    [ -z "$PROMPT" ] && PROMPT="No prompt found"
    SAFE_PROMPT=$(echo "$PROMPT" | sed 's/\.\s*$//' | tr -d '\n')

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



# The prompts used for evaluation

# "A man in black coat, yellow shirt, pink trousers and blue shoes is waving."
# "A man in black coat, yellow shirt, pink trousers, blue leather shoes and green hats is waving."
# "A cartoon girl wears gray shirt, blue skirt, yellow shoes, pink jacket and brown hat is smiling."
# "A cartoon girl with short hair wears gray shirt, blue skirt, yellow shoes, pink jacket and brown hat is dancing."
# "A yellow dog wears a white shirt, two pairs of pink shoes, and a blue hat."
# "A yellow dog wears a pink shirt, two pairs of pink shoes, and a blue collar."
# "A boy wears blue shirt with a yellow star on it, gray trousers, blue sport shoes, purple wizard hat and blue jacket."
# "A boy wears blue shirt with a yellow star on it, gray trousers, blue sport shoes, purple wizard hat and blue jacket, holding a magic stick."
# "A clown with red nose and white face, wears green wig, black shoes, yellow shirt, red jacket, and red pants."

