
import spaces
import os
import gradio as gr
import gc

from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

import glob
import pathlib
from PIL import Image


transform_image = None
birefnet = None

def load_model(model):
    global birefnet
    birefnet = None
    gc.collect()
    torch.cuda.empty_cache()

    birefnet = AutoModelForImageSegmentation.from_pretrained(
        model, trust_remote_code=True
    )
    birefnet.eval()
    birefnet.half()

    spaces.automatically_move_to_gpu_when_forward(birefnet)

with spaces.capture_gpu_object() as birefnet_gpu_obj:
    load_model("ZhengPeng7/BiRefNet_HR")

def common_setup(size):
    global transform_image

    transform_image = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def process(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    image_size = im.size
    image = load_img(im)
    input_image = transform_image(image).unsqueeze(0).to(spaces.gpu).to(torch.float16)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def batch_process(input_folder, output_folder, save_png, save_flat):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', ".avif"]
    
    # Collect all image files from input folder
    input_images = []
    for ext in image_extensions:
        input_images.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
    
    # Process each image
    processed_images = []
    for image_path in input_images:
        try:
            # Load image
            im = load_img(image_path, output_type="pil")
            im = im.convert("RGB")
            image_size = im.size
            image = load_img(im)
            
            # Prepare image for processing
            input_image = transform_image(image).unsqueeze(0).to(spaces.gpu).to(torch.float16)
        
            # Prediction
            with torch.no_grad():
                preds = birefnet(input_image)[-1].sigmoid().cpu()
            
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            
            # Apply mask
            image.putalpha(mask)
            
            # Save processed image
            output_filename = os.path.join(output_folder, f"{pathlib.Path(image_path).name}")

            if save_flat:
                background = Image.new('RGBA', image.size, (255, 255, 255))
                image = Image.alpha_composite(background, image)
                image = image.convert("RGB")
            elif output_filename.lower().endswith(".jpg") or output_filename.lower().endswith(".jpeg"):
                # jpegs don't support alpha channel, so add .png extension (not change, to avoid potential overwrites)
                output_filename += ".png"
            if save_png and not output_filename.lower().endswith(".png"):
                output_filename += ".png"

            image.save(output_filename)
            
            processed_images.append(output_filename)
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    return processed_images


def unload():
    global birefnet, transform_image
    birefnet = None
    transform_image = None
    gc.collect()
    torch.cuda.empty_cache()


css = """
.gradio-container {
    max-width: 1280px !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(css=css, analytics_enabled=False) as demo:
    gr.Markdown("# birefnet for background removal")

    with gr.Tab("image"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an image", type='pil', height=616)
                go_image = gr.Button("Remove background")
            with gr.Column():
                result1 = gr.Image(label="birefnet", type="pil", height=576)

    with gr.Tab("URL"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="URL to image, or local path to image", max_lines=1)
                go_text = gr.Button("Remove background")
            with gr.Column():
                result2 = gr.Image(label="birefnet", type="pil", height=576)

    with gr.Tab("batch"):
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(label="Input folder path", max_lines=1)
                output_dir = gr.Textbox(label="Output folder path (will overwrite)", max_lines=1)
                always_png = gr.Checkbox(label="Always save as PNG", value=True)
                save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                go_batch = gr.Button("Remove background(s)")
            with gr.Column():
                result3 = gr.File(label="Processed image(s)", type="filepath", file_count="multiple")

    with gr.Tab("options"):
        model = gr.Dropdown(label="Model", 
                            choices=["ZhengPeng7/BiRefNet", "ZhengPeng7/BiRefNet_HR"], value="ZhengPeng7/BiRefNet_HR", type="value")
        proc_size = gr.Dropdown(label="birefnet processing image size", info="1024: old model; 2048: HR model - more accurate, uses more VRAM (shared memory works well)", 
                        choices=[1024, 1536, 2048], value=2048)
            
        model.change(fn=load_model, inputs=model, outputs=None)
            
            
    go_image.click(fn=common_setup, inputs=[proc_size]).then(fn=process, inputs=image, outputs=result1)
    go_text.click( fn=common_setup, inputs=[proc_size]).then(fn=process, inputs=text,  outputs=result2)
    go_batch.click(fn=common_setup, inputs=[proc_size]).then(fn=batch_process, inputs=[input_dir, output_dir, always_png, save_flat], outputs=result3)

    demo.unload(unload)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
