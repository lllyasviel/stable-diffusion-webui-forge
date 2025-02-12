
import spaces
import os
import gradio as gr
import gc

try:
    import moviepy.editor as mp
    got_mp = True
except:
    got_mp = False

from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

import glob
import pathlib
from PIL import Image
import numpy


transform_image = None
birefnet = None

def load_model(model):
    global birefnet
    birefnet = None
    gc.collect()
    torch.cuda.empty_cache()

    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/"+model, trust_remote_code=True
    )
    birefnet.eval()
    birefnet.half()

    spaces.automatically_move_to_gpu_when_forward(birefnet)

with spaces.capture_gpu_object() as birefnet_gpu_obj:
    load_model("BiRefNet_HR")

def common_setup(w, h):
    global transform_image

    transform_image = transforms.Compose(
        [
            transforms.Resize((w, h)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def process(image, save_flat, bg_colour):
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

    if save_flat:
        bg_colour += "FF"
        colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
        background = Image.new("RGBA", image_size, colour_rgb)
        image = Image.alpha_composite(background, image)
        image = image.convert("RGB")

    return image

# video processing based on https://huggingface.co/spaces/brokerrobin/video-background-removal/blob/main/app.py
@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def video_process(video, bg_colour):
    # Load the video using moviepy
    video = mp.VideoFileClip(video)

    fps = video.fps

    # Extract audio from the video
    audio = video.audio

    # Extract frames at the specified FPS
    frames = video.iter_frames(fps=fps)

    # Process each frame for background removal
    processed_frames = []

    for i, frame in enumerate(frames):
        print (f"birefnet [video]: frame {i+1}", end='\r', flush=True)

        image = Image.fromarray(frame)
        
        if i == 0:
            image_size = image.size

            colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5))
            background = Image.new("RGBA", image_size, colour_rgb + (255,))
        
        input_image = transform_image(image).unsqueeze(0).to(spaces.gpu).to(torch.float16)
        # Prediction
        with torch.no_grad():
            preds = birefnet(input_image)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)

        # Apply mask and composite
        image.putalpha(mask)
        processed_image = Image.alpha_composite(background, image)

        processed_frames.append(numpy.array(processed_image))

    # Create a new video from the processed frames
    processed_video = mp.ImageSequenceClip(processed_frames, fps=fps)

    # Add the original audio back to the processed video
    processed_video = processed_video.set_audio(audio)

    # Save the processed video using modified original filename (goes to gradio temp)
    filename, _ = os.path.splitext(video.filename)
    filename += "-birefnet.mp4"
    processed_video.write_videofile(filename, codec="libx264")

    return filename


@spaces.GPU(gpu_objects=[birefnet_gpu_obj], manual_load=True)
def batch_process(input_folder, output_folder, save_png, save_flat, bg_colour):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp', ".avif"]
    
    # Collect all image files from input folder
    input_images = []
    for ext in image_extensions:
        input_images.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))

    if save_flat:
        bg_colour += "FF"
        colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
    # Process each image
    processed_images = []
    for i, image_path in enumerate(input_images):
        print (f"birefnet [batch]: image {i+1}", end='\r', flush=True)
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
                background = Image.new("RGBA", image_size, colour_rgb)
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
                image = gr.Image(label="Upload an image", type='pil', height=584)
                go_image = gr.Button("Remove background")
            with gr.Column():
                result1 = gr.Image(label="birefnet", type="pil", height=544)

    with gr.Tab("URL"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="URL to image, or local path to image", max_lines=1)
                go_text = gr.Button("Remove background")
            with gr.Column():
                result2 = gr.Image(label="birefnet", type="pil", height=544)

    if got_mp:
        with gr.Tab("video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Upload a video", height=584)
                    go_video = gr.Button("Remove background")
                with gr.Column():
                    result4 = gr.Video(label="birefnet", height=544, show_share_button=False)

    with gr.Tab("batch"):
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(label="Input folder path", max_lines=1)
                output_dir = gr.Textbox(label="Output folder path (save images will overwrite)", max_lines=1)
                always_png = gr.Checkbox(label="Always save as PNG", value=True)

                go_batch = gr.Button("Remove background(s)")
            with gr.Column():
                result3 = gr.File(label="Processed image(s)", type="filepath", file_count="multiple")

    with gr.Tab("options"):
        gr.Markdown("*HR* : high resolution; *matting* : better with transparency; *lite* : faster.")
        model = gr.Dropdown(label="Model (download on selection, see console for progress)",  
                            choices=["BiRefNet_512x512", "BiRefNet", "BiRefNet_HR", "BiRefNet-matting", "BiRefNet_HR-matting", "BiRefNet_lite", "BiRefNet_lite-2K", "BiRefNet-portrait", "BiRefNet-COD", "BiRefNet-DIS5K", "BiRefNet-DIS5k-TR_TEs", "BiRefNet-HRSOD"], value="BiRefNet_HR", type="value")

        gr.Markdown("Regular models trained at 1024 \u00D7 1024; HR models trained at 2048 \u00D7 2048; 2K model trained at 2560 \u00D7 1440.")
        gr.Markdown("Greater processing image size will typically give more accurate results, but also requires more VRAM (shared memory works well).")
        with gr.Row():
            proc_sizeW = gr.Slider(label="birefnet processing image width",
                            minimum=256, maximum=2560, value=2048, step=32)
            proc_sizeH = gr.Slider(label="birefnet processing image height", 
                            minimum=256, maximum=2048, value=2048, step=32)
        with gr.Row():
            save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
            bg_colour = gr.ColorPicker(label="Background colour for saving flat, and video", value="#00FF00", visible=True, interactive=True)

        model.change(fn=load_model, inputs=model, outputs=None)
            
        gr.Markdown("### https://github.com/ZhengPeng7/BiRefNet\n### https://huggingface.co/ZhengPeng7")

    go_image.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(fn=process, inputs=[image, save_flat, bg_colour], outputs=result1)
    go_text.click( fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(fn=process, inputs=[text, save_flat, bg_colour],  outputs=result2)
    if got_mp:
        go_video.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(
                       fn=video_process, inputs=[video, bg_colour], outputs=result4)
    go_batch.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(
                   fn=batch_process, inputs=[input_dir, output_dir, always_png, save_flat, bg_colour], outputs=result3)

    demo.unload(unload)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
