'''
Code provided by Miaosen Zhang (t-miazhang@microsoft.com)
Call our finetuned Phi-3.5-vision-instruct model with vllm batch inference
'''

import os
from PIL import Image, ImageDraw
import json
from vllm import LLM, SamplingParams


def drawc(imgp, coor):
    image = Image.open(imgp)
    draw = ImageDraw.Draw(image)
    center_x, center_y = coor
    radius = 20
    # Calculate the bounding box of the circle  
    left_up_point = (center_x - radius, center_y - radius)  
    right_down_point = (center_x + radius, center_y + radius)  
    bounding_box = [left_up_point, right_down_point]  
    
    # Draw the empty circle (outline only)  
    draw.ellipse(bounding_box, outline="red", width=10)
    image.save(imgp[:imgp.rfind(".")]+"_output.png")

def process_image(img):
    img_ratio = img.width / img.height  
    target_ratio = (336*3) / (336*2)
    
    if img_ratio > target_ratio:  
        new_width = 336*3  
        new_height = int(new_width / img_ratio) 
    else:  
        new_height = 336*2
        new_width = int(new_height * img_ratio)  
    reshape_ratio = new_width / img.width
    # new_bbox = [int(k*reshape_ratio) for k in bbox]
    img = img.resize((new_width, new_height), Image.LANCZOS)  
    new_img = Image.new("RGB", (336*3, 336*2), (255, 255, 255))  
    paste_position = (0, 0)  
    new_img.paste(img, paste_position)
    return new_img, reshape_ratio

TEMPLATE = """
The description of the element:
{RE}

Locate the above described element in the image. The output should be bounding box using relative coordinates multiplying 1000.
"""

def get_model(model_path):
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=200,
    )
    return llm

def prepare_batch(imgl, rel):
    assert len(imgl) == len(rel)
    prompt_list = []
    image_size_list = []
    reshape_ratio_list = []

    for imgp, re in zip(imgl, rel):
        prompt = "<|user|> \n" + TEMPLATE.format(RE = re) + "<|image_1|> \n<|end|> \n<|assistant|>" 
        img = Image.open(os.path.join(imgp))
        img, reshape_ratio = process_image(img)
        reshape_ratio_list.append(reshape_ratio)
        w,h = img.size
        image_size_list.append((w,h))
        prompt_list.append(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img]
                }
            }
        )
    return prompt_list, image_size_list, reshape_ratio_list

def to_absolute_coordinates(out_strs, imgsl, resl):
    absc = []
    assert len(out_strs) == len(imgsl)
    for out, sz, re in zip(out_strs, imgsl, resl):
        try:
            relative = [float(k)/1000 for k in out.split("<point>")[1].split("</point>")[0].split(', ')]
            absc.append(
                (int(relative[0]*sz[0]/re), int(relative[1]*sz[1]/re))
            )
        except:
            absc.append(f"Error output format for this example, expect <point>x, y</point>, got: {out}")
    return absc


if __name__ == '__main__':
    
    model_path = "microsoft/Phi-Ground"

    llm = get_model(model_path)

    # The batch size here is not limited even to infinity, vllm will dynamically set a batch size.
    image_list = ["./word1.png", "./github.png"]
    reference_list = [
        "When clicking the 'Copilot' button, user can call a powerful AI to fulfill tasks\n" \
        + "The 'Copilot' button is in the top right area of the image, just on the right of the Edictor button.\n" \
        + "The target area is a circle-like icon with gradient color from blue to red, and a 'Copilot' text below the icon.",

        "I want to clone the repository, click the green '<> Code' button."
    ]

    # Suggestions to write reference:
    # Using 3 official types of reference and their combinations: 
    # functional: 'when clicking the target area, user can ...... ' / 'The target area allow user to ...'
    # positional: 'The target area is located at ...... near/next to/between .....'
    # appearance: 'The target area is a ..... color / shapes / icon / ......'
    # using \n to combinate any 1/2/3 of the above, see example 1

    # Or using free form description, see exmaple 2, it will also work. It seems that the model shows certain robustness to the refernece.



    prompt_list, image_size_list, reshape_ratio_list = prepare_batch(image_list, reference_list)


    outputs = llm.generate(prompt_list, sampling_params = SamplingParams(temperature=0, max_tokens=64))
    outs = [o.outputs[0].text for o in outputs]

    absolute_coordinates = to_absolute_coordinates(outs, image_size_list, reshape_ratio_list)
    print(absolute_coordinates)


    # visualization
    for img, coor in zip(image_list, absolute_coordinates):
        if isinstance(coor, str):
            continue
        drawc(img, coor)


