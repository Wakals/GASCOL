import openai
import json, os, pdb, time, sys, json
import pickle
import pandas as pd
import re
import base64
import requests
# from openai.error import RateLimitError
# 目前需要设置代理才可以访问 api
# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

max_retry = 3

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
image_path = "/network_space/server129/qinyiming/GALA3D-main/image_results/man_coat_1019_1/0/60.png"

base64_image = encode_image(image_path)

def get_response(msg):
    response = openai.ChatCompletion.create(
    # response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here is an image"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
            },
            {"type": "text", "text": msg},
        ],
        }
    ],
    max_tokens=300,
    # api_key = "sb-92061fe4f190e8af067b72e654ebb7956dde80a55e4689b9",
    api_key = "sk-mpM0Buym5lAVJG2Y0f8c24B375Df49FeBf4e508f18804bE0",
    )
    return response.choices[0].message['content']


# openai.api_base = "https://api.openai-sb.com/v1"
openai.api_base = "https://29qg.com/v1"


def ImageEvaluate(prompt):

    # ques = "Here is a description of something. Please extract the attributes vocabulary from the description. Then, give me the description without the attributes vocabulary. The description is: " + prompt

    obj = "trousers"
    attr = "pink"

    ques = "Please tell me if the coat of the man is black. If the " + obj + " is totally " + attr + ", please say 'YES' only. If the " + obj + " is not " + attr + ", please say 'NO' only. Don't say anything else."

    res = get_response(ques)

    print(f"res: {res}")

    return res


def get_chain(prompt):
    ques = 'I am now going to use a model to generate text to 3D. This generation is done "inside-to-out", so I need to split the whole sentence into inside-out order as well. Specifically, a sentence contains a subject, instances, and attributes. For example, a yellow dog wearing a blue shirt, a black hat and a red coat is barking. The subject is the yellow dog is barkings, and the instances include shirt, hats, and coat, with the corresponding attributes blue, black, and red. What you need to do is, given a prompt, extract the body and the corresponding instance to the attribute. The next step is to carry out stratification. The rule of stratification is that from inside to outside, if there is a relatively obvious occlusion relationship between instances, then it is necessary to expand one layer down. For example, in the above example, the first layer is shirt and hats. The second layer is the coat. So the stratification order is (blue shirt, black hat, EXTEND, red coat) in which the EXTEND means for the next layer. And the sub-prompt of the first layer is: "A yellow dog wearing a shirt and a hat is barking". And the sub-prompt of the second layer is: "A yellow dog wearing a coat is barking." So what you end up returning is to tell me the body, the instances with corresponding attributes the stratification order and the sub-prompts of each layer. The prompt is: ' + prompt
    ques += '\n'
    ques += 'NOTE: please output a set format of json file. The keys are "body", "instances", "stratification_order", "sub_prompts". And the value of "body" is string, the value of "instances" is a dict, the value of "stratification_order" is a list of strings, and the value of "sub_prompts" is a list of strings.'

    res = get_response(ques)
    # print(f"res: {res}")
    import json
    res = res.strip("```json").strip("```")
    json_dict = json.loads(res)
    return json_dict


if __name__ == "__main__":
    layout = get_chain("A man in black coat, yellow shirt, pink trousers, blue shoes and green hat is waving")
    import json
    layout = layout.strip("```json").strip("```")
    print(layout)
    json_dict = json.loads(layout)
    print(json_dict)
    # layout = get_chain("A girl wears t-shirt, leggings, skirt, hat and shoes is dancing")
