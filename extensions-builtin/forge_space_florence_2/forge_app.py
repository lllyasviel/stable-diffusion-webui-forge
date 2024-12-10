import spaces
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
import os

import requests
import copy

from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random
import numpy as np

# import subprocess
# subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

with spaces.capture_gpu_object() as gpu_object:
    models = {
        # 'microsoft/Florence-2-large-ft': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', attn_implementation='sdpa', trust_remote_code=True).to("cuda").eval(),
        'microsoft/Florence-2-large': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to("cuda").eval(),
        # 'microsoft/Florence-2-base-ft': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True).to("cuda").eval(),
        # 'microsoft/Florence-2-base': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to("cuda").eval(),
    }

    processors = {
        # 'microsoft/Florence-2-large-ft': AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True),
        'microsoft/Florence-2-large': AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True),
        # 'microsoft/Florence-2-base-ft': AutoProcessor.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True),
        # 'microsoft/Florence-2-base': AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True),
    }


DESCRIPTION = "# [Florence-2 Demo](https://huggingface.co/microsoft/Florence-2-large)"

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

@spaces.GPU(gpu_objects=[gpu_object], manual_load=False)
def run_example(task_prompt, image, text_input=None, model_id='microsoft/Florence-2-large'):
    model = models[model_id]
    processor = processors[model_id]
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    ax.axis('off')
    return fig

def draw_polygons(image, prediction, fill_mask=False):

    draw = ImageDraw.Draw(image)
    scale = 1
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue
            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)
    return image

def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  "{}".format(label),
                  align="right",
                  fill=color)
    return image

def process_image(image, task_prompt, text_input=None, model_id='microsoft/Florence-2-large'):
    image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    if task_prompt == 'Caption':
        task_prompt = '<CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'Detailed Caption':
        task_prompt = '<DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'More Detailed Caption':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'Caption + Grounding':
        task_prompt = '<CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        results['<CAPTION>'] = text_input
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Detailed Caption + Grounding':
        task_prompt = '<DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        results['<DETAILED_CAPTION>'] = text_input
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'More Detailed Caption + Grounding':
        task_prompt = '<MORE_DETAILED_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        text_input = results[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        results['<MORE_DETAILED_CAPTION>'] = text_input
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Object Detection':
        task_prompt = '<OD>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<OD>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Dense Region Caption':
        task_prompt = '<DENSE_REGION_CAPTION>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<DENSE_REGION_CAPTION>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Region Proposal':
        task_prompt = '<REGION_PROPOSAL>'
        results = run_example(task_prompt, image, model_id=model_id)
        fig = plot_bbox(image, results['<REGION_PROPOSAL>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Caption to Phrase Grounding':
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        results = run_example(task_prompt, image, text_input, model_id)
        fig = plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
        return results, fig_to_pil(fig)
    elif task_prompt == 'Referring Expression Segmentation':
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        results = run_example(task_prompt, image, text_input, model_id)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
        return results, output_image
    elif task_prompt == 'Region to Segmentation':
        task_prompt = '<REGION_TO_SEGMENTATION>'
        results = run_example(task_prompt, image, text_input, model_id)
        output_image = copy.deepcopy(image)
        output_image = draw_polygons(output_image, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)
        return results, output_image
    elif task_prompt == 'Open Vocabulary Detection':
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = run_example(task_prompt, image, text_input, model_id)
        bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
        fig = plot_bbox(image, bbox_results)
        return results, fig_to_pil(fig)
    elif task_prompt == 'Region to Category':
        task_prompt = '<REGION_TO_CATEGORY>'
        results = run_example(task_prompt, image, text_input, model_id)
        return results, None
    elif task_prompt == 'Region to Description':
        task_prompt = '<REGION_TO_DESCRIPTION>'
        results = run_example(task_prompt, image, text_input, model_id)
        return results, None
    elif task_prompt == 'OCR':
        task_prompt = '<OCR>'
        results = run_example(task_prompt, image, model_id=model_id)
        return results, None
    elif task_prompt == 'OCR with Region':
        task_prompt = '<OCR_WITH_REGION>'
        results = run_example(task_prompt, image, model_id=model_id)
        output_image = copy.deepcopy(image)
        output_image = draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])
        return results, output_image
    else:
        return "", None  # Return empty string and None for unknown task prompts

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""


single_task_list =[
    'Caption', 'Detailed Caption', 'More Detailed Caption', 'Object Detection',
    'Dense Region Caption', 'Region Proposal', 'Caption to Phrase Grounding',
    'Referring Expression Segmentation', 'Region to Segmentation',
    'Open Vocabulary Detection', 'Region to Category', 'Region to Description',
    'OCR', 'OCR with Region'
]

cascased_task_list =[
    'Caption + Grounding', 'Detailed Caption + Grounding', 'More Detailed Caption + Grounding'
]


def update_task_dropdown(choice):
    if choice == 'Cascased task':
        return gr.Dropdown(choices=cascased_task_list, value='Caption + Grounding')
    else:
        return gr.Dropdown(choices=single_task_list, value='Caption')



with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Florence-2 Image Captioning"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                model_selector = gr.Dropdown(choices=list(models.keys()), label="Model", value=list(models.keys())[0])
                task_type = gr.Radio(choices=['Single task', 'Cascased task'], label='Task type selector', value='Single task')
                task_prompt = gr.Dropdown(choices=single_task_list, label="Task Prompt", value="More Detailed Caption")
                task_type.change(fn=update_task_dropdown, inputs=task_type, outputs=task_prompt)
                text_input = gr.Textbox(label="Text Input (optional)")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")
                output_img = gr.Image(label="Output Image")

        gr.Examples(
            examples=[
                [spaces.convert_root_path() + "image1.jpg", "More Detailed Caption"],
                [spaces.convert_root_path() + "image2.jpg", 'OCR with Region']
            ],
            inputs=[input_img, task_prompt],
            outputs=[output_text, output_img],
            fn=process_image,
            cache_examples=False,
            label='Try examples'
        )

        submit_btn.click(process_image, [input_img, task_prompt, text_input, model_selector], [output_text, output_img])


if __name__ == "__main__":
    demo.launch(debug=True)
