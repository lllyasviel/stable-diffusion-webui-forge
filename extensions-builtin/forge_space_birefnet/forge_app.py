
import spaces
import os
import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

import glob
import pathlib
from PIL import Image


with spaces.capture_gpu_object() as birefnet_gpu_obj:
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )

spaces.automatically_move_to_gpu_when_forward(birefnet)

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    image_size = im.size
    origin = im.copy()
    image = load_img(im)
    input_images = transform_image(image).unsqueeze(0).to(spaces.gpu)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return (image, origin)

@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def batch_process(input_folder, output_folder, save_png, save_flat):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
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
            input_image = transform_image(image).unsqueeze(0).to(spaces.gpu)
            
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

slider1 = ImageSlider(label="birefnet", type="pil")
slider2 = ImageSlider(label="birefnet", type="pil")
image = gr.Image(label="Upload an image")
text = gr.Textbox(label="URL to image, or local path to image", max_lines=1)


chameleon = load_img(spaces.convert_root_path() + "chameleon.jpg", output_type="pil")

url = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"
tab1 = gr.Interface(
    fn, inputs=image, outputs=slider1, examples=[chameleon], api_name="image", allow_flagging="never"
)

tab2 = gr.Interface(
    fn, inputs=text, outputs=slider2, examples=[url], api_name="text", allow_flagging="never"
)

tab3 = gr.Interface(
    batch_process, 
    inputs=[
        gr.Textbox(label="Input folder path", max_lines=1),
        gr.Textbox(label="Output folder path (will overwrite)", max_lines=1),
        gr.Checkbox(label="Always save as PNG", value=True),
        gr.Checkbox(label="Save flat (no mask)", value=False)
    ], 
    outputs=gr.File(label="Processed images", type="filepath", file_count="multiple"),
    api_name="batch", 
    allow_flagging="never"
)

demo = gr.TabbedInterface(
    [tab1, tab2, tab3], 
    ["image", "URL", "batch"], 
    title="birefnet for background removal"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
