import math
import time
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Optional OpenCV import for Poisson Blending
try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False
    print("SD Upscale Enhanced: OpenCV not found. Poisson Blending disabled. Install with: pip install opencv-python")


import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, images, devices, sd_samplers # Added sd_samplers for potential future use
from modules.processing import Processed, StableDiffusionProcessingImg2Img
from modules.shared import opts, state
# Attempt to import ControlNet modules/API - THIS IS SPECULATIVE
try:
    # This path might differ based on installation/version
    from scripts import external_code # Common pattern A1111
    controlnet_api = external_code.find_controlnet() # Example function call
    controlnet_version = getattr(controlnet_api, '__version__', 'unknown') if controlnet_api else 'unavailable'
    print(f"SD Upscale Enhanced: Found ControlNet API version {controlnet_version}")
    controlnet_enabled_global = True
except Exception as e:
    print(f"SD Upscale Enhanced: Could not find or import ControlNet API. ControlNet per tile disabled. Error: {e}")
    controlnet_api = None
    controlnet_enabled_global = False


# --- Helper Functions ---

def analyze_tile_variance(tile: Image.Image):
    """Calculates the variance of a PIL Image tile."""
    if tile.mode != 'L':
        tile = tile.convert('L')
    np_tile = np.array(tile)
    variance = np.var(np_tile)
    return variance

def gaussian_weights(size, sigma=1.0):
    """Generate 1D Gaussian weights."""
    x = np.arange(size)
    center = size // 2
    weights = np.exp(-((x - center)**2) / (2 * sigma**2))
    return weights / weights.sum()

def create_blend_mask(size, overlap, direction='horizontal', mode='linear'):
    """Creates a 1D or 2D blend mask (linear or gaussian)."""
    mask = np.ones((size, size))
    weights = None

    if overlap <= 0: return mask # No blending needed

    if mode == 'linear':
        weights = np.linspace(0, 1, overlap)
    elif mode == 'gaussian':
        # Adjust sigma based on overlap size for a reasonable falloff
        sigma = overlap / 4.0 # Heuristic, adjust as needed
        weights = gaussian_weights(overlap * 2, sigma=sigma)[overlap:] # Take half of a symmetric gaussian
        weights = weights / weights.max() # Normalize to range [0, 1]

    if not isinstance(weights, np.ndarray): return mask # Fallback if weights not generated

    if direction == 'horizontal':
        mask[:, :overlap] *= weights
    elif direction == 'vertical':
        mask[:overlap, :] *= weights[:, np.newaxis] # Transpose weights for vertical application

    return mask

def poisson_blend_tile(background, foreground, mask_center):
    """Performs Poisson blending using OpenCV."""
    if not opencv_available:
        print("Error: Poisson blend called but OpenCV is not available.")
        return background # Return original background as fallback

    # Ensure images are in BGR format for OpenCV
    background_bgr = cv2.cvtColor(np.array(background.convert('RGB')), cv2.COLOR_RGB2BGR)
    foreground_bgr = cv2.cvtColor(np.array(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)

    # Create a binary mask (slightly larger than foreground to avoid edge artifacts)
    # Mask should be 1 where foreground pixels are valid
    mask = np.zeros(background_bgr.shape[:2], dtype=np.uint8)
    mask_h, mask_w = mask_center.shape[:2]
    center_y, center_x = background_bgr.shape[0] // 2, background_bgr.shape[1] // 2

    # Calculate top-left corner for placing the foreground mask
    tl_y = center_y - mask_h // 2
    tl_x = center_x - mask_w // 2

    # Ensure mask bounds are within background dimensions
    br_y = min(tl_y + mask_h, background_bgr.shape[0])
    br_x = min(tl_x + mask_w, background_bgr.shape[1])
    crop_h = br_y - tl_y
    crop_w = br_x - tl_x
    
    # Place the mask (assuming mask_center defines the region to blend)
    mask[tl_y:br_y, tl_x:br_x] = mask_center[:crop_h, :crop_w] # Use the provided center mask directly
    
    # Use seamlessClone
    try:
        # Use the center of the background image as the blend center
        blend_center = (center_x, center_y)
        # Blend foreground onto background using the mask
        # Note: Foreground should be the same size as background for this to work correctly
        # If foreground is smaller, adjustments are needed. Assuming they match here.
        if foreground_bgr.shape != background_bgr.shape:
             # Basic center crop/pad if sizes differ (might not be ideal)
             fh, fw = foreground_bgr.shape[:2]
             bh, bw = background_bgr.shape[:2]
             if fh < bh or fw < bw: # Pad foreground
                  pad_y = (bh - fh) // 2
                  pad_x = (bw - fw) // 2
                  fg_padded = cv2.copyMakeBorder(foreground_bgr, pad_y, bh - fh - pad_y, pad_x, bw - fw - pad_x, cv2.BORDER_CONSTANT)
             else: # Crop foreground
                  crop_y = (fh - bh) // 2
                  crop_x = (fw - bw) // 2
                  fg_padded = foreground_bgr[crop_y:crop_y + bh, crop_x:crop_x + bw]
             foreground_bgr = fg_padded

        result_bgr = cv2.seamlessClone(foreground_bgr, background_bgr, mask, blend_center, cv2.NORMAL_CLONE) # Or MIXED_CLONE
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    except cv2.error as e:
        print(f"Error during Poisson Blending: {e}. Falling back to simple paste.")
        # Fallback: simple paste using the mask (less seamless)
        result_bgr = background_bgr.copy()
        mask_indices = np.where(mask == 255)
        result_bgr[mask_indices] = foreground_bgr[mask_indices]
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

def combine_grid_custom(grid, blend_mode='simple_overlap', overlap=64):
    """Combines tiles with specified blending."""
    if blend_mode == 'simple_overlap' or overlap == 0:
        # Use existing simple combine if no blending or simple overlap
        return images.combine_grid(grid)

    # --- Advanced Blending Logic ---
    print(f"Combining grid with '{blend_mode}' blending (overlap={overlap})...")
    
    # Get dimensions
    rows = len(grid.tiles)
    cols = len(grid.tiles[0][2]) if rows > 0 else 0
    if rows == 0 or cols == 0: return Image.new("RGB", (0,0))

    tile_w = grid.tile_w
    tile_h = grid.tile_h
    image_w = grid.image_w
    image_h = grid.image_h

    # Create the target image
    result = Image.new("RGB", (image_w, image_h))

    # Store processed tiles in a 2D list for easier access
    processed_tiles = [[grid.tiles[y][2][x][2] for x in range(cols)] for y in range(rows)]

    # --- Poisson Blending (Special Case) ---
    if blend_mode == 'poisson' and opencv_available:
        print("Applying Poisson Blending (may be slow)...")
        # Poisson blending is complex to apply perfectly to a grid.
        # A simpler approach is to blend each tile onto a canvas sequentially.
        # This won't be as good as a global solve but is more feasible.
        
        # Create a mask representing the center non-overlapping part of a tile
        center_mask = np.ones((tile_h - overlap*2, tile_w - overlap*2), dtype=np.uint8) * 255 # Mask for the unique center
        # Pad mask to full tile size
        full_mask = cv2.copyMakeBorder(center_mask, overlap, overlap, overlap, overlap, cv2.BORDER_CONSTANT, value=0)
        
        current_image_np = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        
        for y in range(rows):
            for x in range(cols):
                print(f"Blending tile ({y},{x}) using Poisson...")
                tile_img = processed_tiles[y][x]
                paste_x = grid.tiles[y][2][x][0]
                paste_y = grid.tiles[y][2][x][1]
                
                # Define background region (slightly larger than tile to provide context)
                bg_x1 = max(0, paste_x - overlap // 2)
                bg_y1 = max(0, paste_y - overlap // 2)
                bg_x2 = min(image_w, paste_x + tile_w + overlap // 2)
                bg_y2 = min(image_h, paste_y + tile_h + overlap // 2)
                background_region = current_image_np[bg_y1:bg_y2, bg_x1:bg_x2]
                
                # Prepare foreground (current tile) and mask, resizing if needed
                fg_region = cv2.cvtColor(np.array(tile_img), cv2.COLOR_RGB2BGR)
                mask_region = full_mask
                
                # Calculate center point relative to background_region
                center_x_rel = paste_x - bg_x1 + tile_w // 2
                center_y_rel = paste_y - bg_y1 + tile_h // 2
                
                # Ensure fg_region and mask match size (crop if necessary)
                fg_region = fg_region[:background_region.shape[0], :background_region.shape[1]]
                mask_region = mask_region[:background_region.shape[0], :background_region.shape[1]]
                
                # Perform blending
                try:
                    blended_region = cv2.seamlessClone(fg_region, background_region, mask_region, (center_x_rel, center_y_rel), cv2.NORMAL_CLONE)
                    current_image_np[bg_y1:bg_y2, bg_x1:bg_x2] = blended_region
                except cv2.error as e:
                     print(f" CV2 Error blending tile ({y},{x}): {e}. Pasting directly.")
                     # Fallback paste
                     current_image_np[paste_y:paste_y+tile_h, paste_x:paste_x+tile_w] = fg_region[:tile_h, :tile_w] # Paste only the tile area

        return Image.fromarray(cv2.cvtColor(current_image_np, cv2.COLOR_BGR2RGB))
        # --- End Poisson Placeholder ---


    # --- Linear / Gaussian Blending ---
    # Use numpy for faster pixel manipulation
    result_np = np.array(result).astype(np.float32)
    contributors = np.zeros_like(result_np) # To average contributions in overlap areas

    blend_type = 'linear' if blend_mode == 'linear' else 'gaussian'

    for y in range(rows):
        for x in range(cols):
            tile_img = processed_tiles[y][x]
            tile_np = np.array(tile_img).astype(np.float32)
            paste_x = grid.tiles[y][2][x][0]
            paste_y = grid.tiles[y][2][x][1]

            # Create blend weights for this tile
            # Weight is 1 in center, falls off in overlap regions
            weights_np = np.ones_like(tile_np)

            # Apply vertical blending weights (top overlap)
            if y > 0:
                mask = create_blend_mask(overlap, overlap, direction='vertical', mode=blend_type)
                weights_np[:overlap, :, :] *= mask[:, :, np.newaxis] # Expand mask to 3 channels
            # Apply vertical blending weights (bottom overlap)
            if y < rows - 1:
                 mask = create_blend_mask(overlap, overlap, direction='vertical', mode=blend_type)
                 weights_np[tile_h-overlap:, :, :] *= (1.0 - mask[::-1, :, np.newaxis]) # Inverted mask

            # Apply horizontal blending weights (left overlap)
            if x > 0:
                mask = create_blend_mask(overlap, overlap, direction='horizontal', mode=blend_type)
                weights_np[:, :overlap, :] *= mask[:, :, np.newaxis]
            # Apply horizontal blending weights (right overlap)
            if x < cols - 1:
                 mask = create_blend_mask(overlap, overlap, direction='horizontal', mode=blend_type)
                 weights_np[:, tile_w-overlap:, :] *= (1.0 - mask[:, ::-1, np.newaxis])

            # Add weighted tile to result
            result_np[paste_y:paste_y+tile_h, paste_x:paste_x+tile_w, :] += tile_np * weights_np
            contributors[paste_y:paste_y+tile_h, paste_x:paste_x+tile_w, :] += weights_np

    # Normalize by the number of contributors (avoid division by zero)
    result_np = np.divide(result_np, contributors, out=np.zeros_like(result_np), where=contributors != 0)
    result_np = np.clip(result_np, 0, 255) # Clip values

    return Image.fromarray(result_np.astype(np.uint8))


# --- Main Script Class ---

class Script(scripts.Script):
    def title(self):
        return "SD upscale Enhanced" # Updated title

    def show(self, is_img2img):
        return is_img2img # Only show in img2img

    def ui(self, is_img2img):
        with gr.Blocks():
            gr.HTML("<p style=\"margin-bottom:0.25em\">Enhanced tiled upscale using SD. Uses main img2img settings (prompt, denoising, sampler, steps, CN etc.) on tiles.</p><p style=\"margin-bottom:0.75em\">Set <b>Tile Size</b> using the main Width/Height sliders below. Input image mask is used if 'Masked SD Processing' is checked.</p>")

            with gr.Row():
                scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Initial Upscale Factor', value=2.0, elem_id=self.elem_id("scale_factor"))
                upscaler_index = gr.Radio(label='Initial Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", elem_id=self.elem_id("upscaler_index"))

            with gr.Row():
                overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap (px)', value=64, elem_id=self.elem_id("overlap"))
                blend_mode = gr.Dropdown(label="Seam Blending", choices=["Simple Overlap", "Linear Blend", "Gaussian Blend"] + (["Poisson Blend"] if opencv_available else []), value="Simple Overlap", elem_id=self.elem_id("blend_mode"))

            with gr.Accordion("Advanced Processing Options", open=False):
                 with gr.Row():
                     use_mask = gr.Checkbox(label="Masked SD Processing", value=False, elem_id=self.elem_id("use_mask"), tooltip="Only process tiles overlapping the input img2img mask with SD.")
                     skip_flat = gr.Checkbox(label="Skip SD for flat tiles", value=False, elem_id=self.elem_id("skip_flat"))
                     flat_threshold = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, label='Flatness Threshold (Variance)', value=10.0, elem_id=self.elem_id("flat_threshold"), visible=False) # Show only if skip_flat checked
                     skip_flat.change(fn=lambda x: gr.update(visible=x), inputs=skip_flat, outputs=flat_threshold)

                 with gr.Row():
                     use_cn = gr.Checkbox(label="Enable ControlNet per Tile", value=False, elem_id=self.elem_id("use_cn"), visible=controlnet_enabled_global, tooltip="Use original image region as input for active ControlNets on each tile.")
                     # Add placeholder for selecting which CN unit(s) if needed later
                     cn_info_text = gr.HTML("<p style='color:grey; font-size:0.8em'>Uses active ControlNet settings from main UI. Ensure CN model is suitable for guidance (e.g., Tile, Canny).</p>", visible=False)
                     use_cn.change(fn=lambda x: gr.update(visible=x), inputs=use_cn, outputs=cn_info_text)


                 with gr.Row():
                    use_downscale_pass = gr.Checkbox(label="Apply Downscaling Pre-pass", value=False, elem_id=self.elem_id("use_downscale"))
                    downscale_factor = gr.Slider(minimum=0.25, maximum=1.0, step=0.05, label='Pre-pass Downscale Factor', value=0.75, elem_id=self.elem_id("downscale_factor"), visible=False)
                    use_downscale_pass.change(fn=lambda x: gr.update(visible=x), inputs=use_downscale_pass, outputs=downscale_factor)

            with gr.Accordion("Preview", open=False):
                 preview_button = gr.Button("Preview Tiling")
                 tile_preview_image = gr.Image(label="Tiling Preview", interactive=False, show_label=True)


            # --- Preview Handler ---
            def generate_tile_preview(img, tile_w, tile_h, tile_overlap):
                 if img is None:
                     return None
                 
                 # Ensure img is PIL Image
                 if not isinstance(img, Image.Image):
                      # Try getting from dict if it's Gradio input format
                      if isinstance(img, dict) and 'image' in img and isinstance(img['image'], Image.Image):
                           img = img['image']
                      else:
                           # If it's a numpy array or other, try to convert
                           try: img = Image.fromarray(img)
                           except Exception: return None # Cannot process

                 preview_img = img.copy()
                 max_preview_size = 512 # Limit preview size
                 preview_img.thumbnail((max_preview_size, max_preview_size))
                 draw = ImageDraw.Draw(preview_img)
                 
                 scale_x = preview_img.width / img.width
                 scale_y = preview_img.height / img.height

                 grid = images.split_grid(img, tile_w=tile_w, tile_h=tile_h, overlap=tile_overlap)

                 for y, h, row in grid.tiles:
                     for x, w, _tile_img_data in row:
                         # Draw rectangle on the scaled preview
                         draw.rectangle([
                             x * scale_x,
                             y * scale_y,
                             (x + w) * scale_x -1, # -1 for visibility
                             (y + h) * scale_y -1
                         ], outline="red", width=1)
                 
                 return preview_img

            # Need access to the main img2img components (image, width, height) for preview
            # This requires finding them in the Gradio Blocks tree, which is complex from script code.
            # Placeholder: Assume user triggers preview *after* selecting image and setting sliders.
            # We need the main 'width' and 'height' sliders from the core UI.
            # This might need adjustment based on the exact structure of the UI.
            try:
                # Attempt to find components by elem_id (fragile)
                img2img_image_component = self.gradio_components["img2img_image"] # Hypothetical ID
                img2img_w_slider = self.gradio_components["img2img_width"] # Hypothetical ID
                img2img_h_slider = self.gradio_components["img2img_height"] # Hypothetical ID
                
                preview_button.click(
                    fn=generate_tile_preview,
                    inputs=[img2img_image_component, img2img_w_slider, img2img_h_slider, overlap],
                    outputs=[tile_preview_image]
                )
            except Exception as e:
                 print(f"SD Upscale Enhanced: Failed to link preview button - UI component IDs might be incorrect. Error: {e}")
                 # Disable preview button if linking fails
                 preview_button.interactive = False # Disable if we can't find components


        # The list of UI elements returned by ui() determines the arguments received by run()
        return [overlap, upscaler_index, scale_factor, blend_mode, use_mask, skip_flat, flat_threshold, use_cn, use_downscale_pass, downscale_factor]


    def run(self, p: StableDiffusionProcessingImg2Img, overlap, upscaler_index, scale_factor, blend_mode, use_mask, skip_flat, flat_threshold, use_cn, use_downscale_pass, downscale_factor):

        start_time = time.time()
        print(f"Starting SD Upscale Enhanced run...")

        # --- Input Validation & Setup ---
        if not p.init_images:
            print("SD upscale error: No initial image found in p.init_images.")
            return Processed(p, [], p.seed, "No initial image")
        
        if scale_factor <= 1.0 and upscaler_index != "None": # Check if upscaler_index is name or index
             if isinstance(upscaler_index, int) and shared.sd_upscalers[upscaler_index].name == "None":
                 pass # Allow None upscaler even if scale is 1
             elif isinstance(upscaler_index, str) and upscaler_index == "None":
                 pass
             else:
                  print("SD upscale warning: Scale factor is <= 1.0, but an upscaler is selected. The upscaler will be skipped.")
                  upscaler_index = "None" # Force disable upscaler

        # Ensure upscaler_index is an index
        if isinstance(upscaler_index, str):
            try:
                upscaler_name_lower = upscaler_index.lower()
                upscaler_index = next(i for i, u in enumerate(shared.sd_upscalers) if u.name.lower() == upscaler_name_lower)
            except StopIteration:
                 print(f"SD upscale error: Upscaler '{upscaler_index}' not found. Using None.")
                 upscaler_index = next(i for i, u in enumerate(shared.sd_upscalers) if u.name == "None") # Find index of None

        processing.fix_seed(p) # Ensure seed is fixed if -1
        upscaler = shared.sd_upscalers[upscaler_index]
        tile_w, tile_h = p.width, p.height # Tile size from main UI sliders

        print(f" Tile size: {tile_w}x{tile_h}, Overlap: {overlap}, Initial Upscaler: {upscaler.name}, Scale Factor: {scale_factor}")
        print(f" Blend Mode: {blend_mode}, Masked: {use_mask}, Skip Flat: {skip_flat} (Thr: {flat_threshold}), CN: {use_cn}, Downscale Pass: {use_downscale_pass} (Factor: {downscale_factor})")

        # Store extra parameters
        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = upscaler.name
        p.extra_generation_params["SD upscale factor"] = scale_factor
        p.extra_generation_params["SD upscale blend mode"] = blend_mode
        p.extra_generation_params["SD upscale masked"] = use_mask
        p.extra_generation_params["SD upscale skip flat"] = f"{skip_flat} (Thr: {flat_threshold})"
        p.extra_generation_params["SD upscale use_cn"] = use_cn
        p.extra_generation_params["SD upscale downscale_pass"] = f"{use_downscale_pass} (Factor: {downscale_factor})"

        initial_info = None
        seed = p.seed # Store initial seed

        # --- Initial Image Preparation ---
        init_img_original = p.init_images[0].copy() # Keep original for ControlNet/Masking
        init_img_processed = images.flatten(init_img_original, opts.img2img_background_color) # Apply background color flatten

        # --- 1. Initial Upscale ---
        if upscaler.name != "None" and scale_factor > 1.0:
            print(f"Performing initial upscale with {upscaler.name}...")
            try:
                upscaled_img = upscaler.scaler.upscale(init_img_processed, scale_factor, upscaler.data_path)
            except Exception as e:
                print(f"Error during initial upscale: {e}. Using original image.")
                upscaled_img = init_img_processed # Fallback to original if upscale fails
        else:
            print("Skipping initial upscale.")
            upscaled_img = init_img_processed # Use original if no upscale needed

        devices.torch_gc()
        print(f"Upscaled image size: {upscaled_img.width}x{upscaled_img.height}")

        # --- Mask Preparation (if needed) ---
        mask_img = None
        if use_mask and p.image_mask:
            print("Preparing mask for masked processing...")
            mask_img = p.image_mask.convert('L').resize((upscaled_img.width, upscaled_img.height), Image.NEAREST)
            mask_np = np.array(mask_img) # Numpy mask for faster checks

        # --- 2. Tiling and Tile Preparation ---
        print(f"Splitting into {tile_w}x{tile_h} tiles with {overlap}px overlap...")
        grid = images.split_grid(upscaled_img, tile_w=tile_w, tile_h=tile_h, overlap=overlap)

        # Store original upscaled tiles for skipping/masking/blending reference
        original_upscaled_tiles = {} # {(x,y): tile_image}

        work = [] # List of tiles to process with SD
        skipped_tiles = {} # {(x,y): original_upscaled_tile} - Store skipped tiles' original content

        total_tile_count = 0
        for y_coord, h, row in grid.tiles:
            for x_coord, w, tile_img_data in row:
                total_tile_count += 1
                tile_coords = (x_coord, y_coord)
                tile_img = tile_img_data # This is the actual tile image
                original_upscaled_tiles[tile_coords] = tile_img.copy() # Store copy

                process_this_tile = True

                # Check mask
                if use_mask and mask_img:
                    # Check if tile bounding box overlaps with non-black area in mask_np
                    tile_mask_region = mask_np[y_coord:y_coord+h, x_coord:x_coord+w]
                    if not np.any(tile_mask_region > 0): # Check if any pixel in the mask region is > 0
                        process_this_tile = False
                        # print(f" Skipping tile ({x_coord},{y_coord}) due to mask.")

                # Check flatness (only if not already skipped by mask)
                if process_this_tile and skip_flat:
                    variance = analyze_tile_variance(tile_img)
                    if variance < flat_threshold:
                        process_this_tile = False
                        # print(f" Skipping tile ({x_coord},{y_coord}) due to flatness (Variance: {variance:.2f} < {flat_threshold}).")

                # Add to work list or skipped list
                if process_this_tile:
                    work.append({'image': tile_img, 'coords': tile_coords})
                else:
                    skipped_tiles[tile_coords] = tile_img # Store the original tile

        print(f"Total tiles: {total_tile_count}. Tiles to process with SD: {len(work)}. Skipped tiles: {len(skipped_tiles)}.")

        if not work:
            print("SD upscale warning: No tiles left to process after masking/skipping. Returning initially upscaled image.")
            # Need to save the upscaled_img manually if opts.samples_save is True
            if opts.samples_save:
                 images.save_image(upscaled_img, p.outpath_samples, "", seed, p.prompt, opts.samples_format, info=p.info, p=p)
            return Processed(p, [upscaled_img], seed, "No tiles processed with SD.")

        # --- 3. SD Processing in Batches ---
        batch_size = p.batch_size
        upscale_count = p.n_iter # Number of times to upscale the whole image
        p.n_iter = 1 # We handle iterations manually
        p.do_not_save_grid = True # Grid is combined manually
        p.do_not_save_samples = True # Samples are saved manually

        batch_count = math.ceil(len(work) / batch_size)
        state.job_count = batch_count * upscale_count
        print(f"Processing {len(work)} tiles in {batch_count} batches per upscale ({upscale_count} total upscales)...")

        processed_tiles_all_runs = [] # List of dictionaries: [{'coords': (x,y), 'image': img}, ...] per run

        # --- ControlNet Integration Placeholder ---
        # Store original ControlNet settings if modifying p directly
        original_cn_script_args = None
        cn_units = []
        if use_cn and controlnet_enabled_global:
            print("Preparing ControlNet for tiled processing...")
            # --- !!! This part needs adaptation based on Web UI/CN version !!! ---
            # Find the ControlNet script arguments in p.script_args or dedicated attribute
            # Example: Find by script title (highly fragile)
            cn_script = None
            if hasattr(p, 'scripts') and p.scripts and hasattr(p.scripts, 'scripts'):
                 for script_obj in p.scripts.scripts:
                      if 'controlnet' in script_obj.title().lower():
                           cn_script = script_obj
                           break
            
            if cn_script and hasattr(cn_script, 'args_from') and hasattr(cn_script, 'args_to'):
                 print(f" Found potential ControlNet script: {cn_script.title()}")
                 # Store original args
                 original_cn_script_args = p.script_args[cn_script.args_from:cn_script.args_to]
                 # Parse args into ControlNet units (assuming external_code structure)
                 # This structure is complex and version-dependent. This is a GUESS.
                 num_units = opts.data.get("control_net_max_models_num", 1)
                 try:
                     # This parsing logic is likely incorrect/incomplete.
                     # Need to accurately map script args back to ControlNetUnit objects or dicts.
                     parsed_units = external_code.parse_controlnet_args_legacy(p, original_cn_script_args) # Example hypothetical parsing
                     cn_units = [unit for unit in parsed_units if unit.enabled and unit.image is not None] # Use only enabled units with an initial image placeholder
                     print(f" Parsed {len(cn_units)} enabled ControlNet units to modify.")
                 except Exception as cn_parse_err:
                      print(f" Error parsing ControlNet args (may need script update for your CN version): {cn_parse_err}")
                      use_cn = False # Disable if parsing fails
            elif hasattr(p, 'control_net_units'): # Alternative common pattern
                 print(" Found ControlNet units in p.control_net_units")
                 # Need to handle deepcopy or careful modification
                 # Store original images from units that will be modified
                 original_cn_unit_images = {}
                 cn_units = []
                 for i, unit in enumerate(p.control_net_units):
                     if unit.enabled: # Check if the unit is actually enabled
                          cn_units.append({'index': i, 'original_image': unit.image.copy() if unit.image else None}) # Store index and original image
                 print(f" Stored original images for {len(cn_units)} enabled ControlNet units.")
                 # If no enabled units, disable the feature for this run
                 if not cn_units: use_cn = False
            else:
                 print(" Could not find ControlNet arguments in p object. Disabling CN per tile.")
                 use_cn = False
        # --- End ControlNet Placeholder ---


        for n in range(upscale_count):
            print(f"--- Upscale Run {n + 1}/{upscale_count} ---")
            start_seed_run = seed + n
            p.seed = start_seed_run # Set seed for the start of this upscale run

            processed_tiles_current_run = {} # {(x,y): image} for this run
            work_index = 0

            for i in range(batch_count):
                if state.interrupted: break
                
                batch_start_index = i * batch_size
                batch_end_index = min((i + 1) * batch_size, len(work))
                batch_work_items = work[batch_start_index:batch_end_index]

                # Prepare batch images
                p.batch_size = len(batch_work_items) # Adjust batch size for last batch
                batch_init_images = [item['image'] for item in batch_work_items]

                # --- Downscaling Pre-pass ---
                original_batch_sizes = []
                if use_downscale_pass and downscale_factor < 1.0:
                    downscaled_batch = []
                    for img in batch_init_images:
                         original_batch_sizes.append(img.size)
                         w, h = img.size
                         target_w = int(w * downscale_factor)
                         target_h = int(h * downscale_factor)
                         downscaled_batch.append(img.resize((target_w, target_h), Image.LANCZOS))
                    p.init_images = downscaled_batch
                    # Optionally adjust p.width/p.height? Might confuse process_images. Test this.
                    # print(f" Downscaled batch to ~{target_w}x{target_h} for processing")
                else:
                    p.init_images = batch_init_images

                # --- Apply ControlNet Images for Batch ---
                if use_cn and cn_units:
                    try:
                         # Iterate through tiles in the current batch
                         for batch_idx, work_item in enumerate(batch_work_items):
                             tile_coords = work_item['coords']
                             x_coord, y_coord = tile_coords
                             
                             # Find corresponding region in original low-res image
                             # Map tile coords back to original image coords
                             orig_x = int(x_coord / scale_factor)
                             orig_y = int(y_coord / scale_factor)
                             orig_w = int(tile_w / scale_factor)
                             orig_h = int(tile_h / scale_factor)
                             
                             # Ensure bounds are valid
                             orig_x = max(0, orig_x)
                             orig_y = max(0, orig_y)
                             orig_w = min(init_img_original.width - orig_x, orig_w)
                             orig_h = min(init_img_original.height - orig_y, orig_h)

                             if orig_w <= 0 or orig_h <= 0: continue # Skip if region is invalid

                             control_image_source = init_img_original.crop((orig_x, orig_y, orig_x + orig_w, orig_y + orig_h))

                             # Apply this source image to the enabled ControlNet units FOR THIS BATCH ITEM
                             # This requires modifying the correct part of p.script_args or p.control_net_units
                             # Assuming p.control_net_units structure:
                             if hasattr(p, 'control_net_units'):
                                  # Modify units *for this specific image in the batch*
                                  # ControlNet extension needs to support per-image inputs in a batch, or we process batch_size=1
                                  # If processing batch_size > 1, this direct modification might apply the LAST tile's CN to the whole batch.
                                  # SAFEST APPROACH: Force batch size 1 if using per-tile CN.
                                  if p.batch_size > 1:
                                       print("Warning: Forcing batch size to 1 for ControlNet per Tile accuracy.")
                                       # This requires re-batching logic, major change.
                                       # Simpler alternative: Assume CN extension handles list of control images if p.init_images is a list.
                                       # This depends entirely on CN implementation. Let's try setting it per unit index.
                                       
                                  for unit_info in cn_units:
                                       unit_index = unit_info['index']
                                       if unit_index < len(p.control_net_units):
                                           # Prepare image for unit (resize etc. if needed by CN)
                                           # control_image_prepared = controlnet_api.prepare_image(control_image_source, ...) # Hypothetical API call
                                           control_image_prepared = control_image_source # Simple assignment
                                           
                                           # Store the image directly in the unit for this process call
                                           # Need a way to pass a LIST of control images if batching
                                           # This simple assignment likely only works for batch size 1
                                           p.control_net_units[unit_index].image = control_image_prepared


                             # --- Else if using script_args (more complex) ---
                             # else:
                             #   Find correct slice in p.script_args for CN image parameter for the relevant unit(s)
                             #   Modify p.script_args[correct_index] = control_image_source
                             #   This is extremely fragile.

                    except Exception as cn_apply_err:
                         print(f" Error applying ControlNet image for tile {tile_coords}: {cn_apply_err}. Skipping CN for this batch.")
                # --- End ControlNet Application ---


                # --- Process Batch ---
                state.job = f"Batch {i + 1 + n * batch_count}/{state.job_count}"
                # print(f" Processing batch {i+1}/{batch_count} (Run {n+1})")

                try:
                    processed: Processed = processing.process_images(p)
                except Exception as batch_err:
                    print(f"Error processing batch {i+1}: {batch_err}")
                    # Fill results with placeholders or skip? Fill with black for now.
                    processed = Processed(p, [Image.new("RGB", (tile_w, tile_h), color="black")] * p.batch_size, p.seed, f"Error: {batch_err}")

                if initial_info is None: # Save info from the first successful batch
                    initial_info = processed.info

                # --- Upscale Result (if downscaling pre-pass was used) ---
                batch_results = []
                if use_downscale_pass and downscale_factor < 1.0:
                    # print(f" Upscaling processed batch back to {tile_w}x{tile_h}")
                    # Use a fast, decent quality upscaler like Lanczos or maybe the selected one
                    upscaler_to_use = Image.LANCZOS # Or find selected upscaler object
                    for idx, img in enumerate(processed.images):
                         original_w, original_h = original_batch_sizes[idx]
                         batch_results.append(img.resize((original_w, original_h), upscaler_to_use))
                else:
                     batch_results = processed.images

                # Assign processed tiles back to their coordinates for the current run
                for idx, item in enumerate(batch_work_items):
                    if idx < len(batch_results):
                        processed_tiles_current_run[item['coords']] = batch_results[idx]
                    work_index += 1

                # Restore original ControlNet images if they were modified directly
                if use_cn and cn_units and hasattr(p, 'control_net_units'):
                     for unit_info in cn_units:
                          if unit_info['index'] < len(p.control_net_units):
                               p.control_net_units[unit_info['index']].image = unit_info['original_image']

                # Update seed for next batch (consistent progression)
                p.seed = processed.seed + 1
                devices.torch_gc() # Garbage collect VRAM

            # --- End of Batch Loop for Run ---
            if state.interrupted: break
            processed_tiles_all_runs.append(processed_tiles_current_run)

        # --- End of Upscale Runs Loop ---

        if state.interrupted:
             print("SD upscale interrupted.")
             # Return potentially incomplete results?
             if not processed_tiles_all_runs: return Processed(p, [], seed, "Interrupted before completion.")
             # Use the last completed run's tiles
             processed_tiles_to_combine = processed_tiles_all_runs[-1]
        elif not processed_tiles_all_runs:
             print("SD upscale error: No processing runs completed successfully.")
             return Processed(p, [], seed, "No processing runs completed.")
        else:
             # Use the tiles from the last run for combination
             processed_tiles_to_combine = processed_tiles_all_runs[-1]


        # --- 4. Recombine Tiles ---
        print("Recombining processed tiles...")
        image_index = 0
        # Fill the grid structure with processed tiles and skipped tiles
        for y_coord, h, row in grid.tiles:
            for x_coord, w, _ in row:
                tile_coords = (x_coord, y_coord)
                if tile_coords in processed_tiles_to_combine:
                     grid.tiles[y_coord//h][2][x_coord//w][2] = processed_tiles_to_combine[tile_coords]
                elif tile_coords in skipped_tiles:
                     grid.tiles[y_coord//h][2][x_coord//w][2] = skipped_tiles[tile_coords]
                else:
                     # Should not happen if logic is correct, but fallback
                     print(f"Warning: Tile at ({x_coord},{y_coord}) not found in processed or skipped lists. Using black.")
                     grid.tiles[y_coord//h][2][x_coord//w][2] = Image.new("RGB", (w, h), color="black")

        # Use the custom combine function
        combined_image = combine_grid_custom(grid, blend_mode=blend_mode, overlap=overlap)
        print("Tile recombination complete.")

        # --- Final Output and Saving ---
        # We only output the result of the last run in this structure
        # If multiple runs were needed, they'd ideally be saved individually
        # For now, return the last combined image as the primary result.
        result_images = [combined_image]

        # Manually save the final combined image(s) if requested
        # (Currently saves only the last run's result)
        if opts.samples_save:
            final_seed = seed + (len(processed_tiles_all_runs) -1) if processed_tiles_all_runs else seed
            print(f"Saving final upscaled image (seed: {final_seed})...")
            # Ensure initial_info is not None before saving
            if initial_info is None:
                 initial_info = f"SD Upscale Enhanced parameters: {p.extra_generation_params}" # Basic fallback info
            images.save_image(combined_image, p.outpath_samples, "", final_seed, p.prompt, opts.samples_format, info=initial_info, p=p)

        # Restore original args if modified
        # if use_cn and original_cn_script_args is not None and cn_script:
        #      p.script_args = p.script_args[:cn_script.args_from] + original_cn_script_args + p.script_args[cn_script.args_to:]
        # Restore ControlNet units if modified directly (already done in loop?)
        if use_cn and cn_units and hasattr(p, 'control_net_units'):
             for unit_info in cn_units:
                  if unit_info['index'] < len(p.control_net_units):
                       p.control_net_units[unit_info['index']].image = unit_info['original_image']


        # Restore original p settings changed by the script
        p.n_iter = upscale_count # Restore original n_iter
        p.batch_size = batch_size # Restore original batch_size (might have been changed)
        p.do_not_save_grid = False
        p.do_not_save_samples = False # Allow normal saving for subsequent operations if any

        end_time = time.time()
        print(f"SD Upscale Enhanced finished in {end_time - start_time:.2f} seconds.")

        return Processed(p, result_images, seed, initial_info) # Return results from last run
