import numpy as np
import gradio as gr
import math
from modules.ui_components import InputAccordion
import modules.scripts as scripts
from modules.torch_utils import float64

from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import convolve
from joblib import Parallel, delayed, cpu_count

class SoftInpaintingSettings:
    def __init__(self,
                 mask_blend_power,
                 mask_blend_scale,
                 inpaint_detail_preservation,
                 composite_mask_influence,
                 composite_difference_threshold,
                 composite_difference_contrast):
        self.mask_blend_power = mask_blend_power
        self.mask_blend_scale = mask_blend_scale
        self.inpaint_detail_preservation = inpaint_detail_preservation
        self.composite_mask_influence = composite_mask_influence
        self.composite_difference_threshold = composite_difference_threshold
        self.composite_difference_contrast = composite_difference_contrast

    def add_generation_params(self, dest):
        dest[enabled_gen_param_label] = True
        dest[gen_param_labels.mask_blend_power] = self.mask_blend_power
        dest[gen_param_labels.mask_blend_scale] = self.mask_blend_scale
        dest[gen_param_labels.inpaint_detail_preservation] = self.inpaint_detail_preservation
        dest[gen_param_labels.composite_mask_influence] = self.composite_mask_influence
        dest[gen_param_labels.composite_difference_threshold] = self.composite_difference_threshold
        dest[gen_param_labels.composite_difference_contrast] = self.composite_difference_contrast


# ------------------- Methods -------------------

def processing_uses_inpainting(p):
    # TODO: Figure out a better way to determine if inpainting is being used by p
    if getattr(p, "image_mask", None) is not None:
        return True

    if getattr(p, "mask", None) is not None:
        return True

    if getattr(p, "nmask", None) is not None:
        return True

    return False


def latent_blend(settings, a, b, t):
    """
    Interpolates two latent image representations according to the parameter t,
    where the interpolated vectors' magnitudes are also interpolated separately.
    The "detail_preservation" factor biases the magnitude interpolation towards
    the larger of the two magnitudes.
    """
    import torch

    # NOTE: We use inplace operations wherever possible.

    if len(t.shape) == 3:
        # [4][w][h] to [1][4][w][h]
        t2 = t.unsqueeze(0)
        # [4][w][h] to [1][1][w][h] - the [4] seem redundant.
        t3 = t[0].unsqueeze(0).unsqueeze(0)
    else:
        t2 = t
        t3 = t[:, 0][:, None]

    one_minus_t2 = 1 - t2
    one_minus_t3 = 1 - t3

    # Linearly interpolate the image vectors.
    a_scaled = a * one_minus_t2
    b_scaled = b * t2
    image_interp = a_scaled
    image_interp.add_(b_scaled)
    result_type = image_interp.dtype
    del a_scaled, b_scaled, t2, one_minus_t2

    # Calculate the magnitude of the interpolated vectors. (We will remove this magnitude.)
    # 64-bit operations are used here to allow large exponents.
    current_magnitude = torch.norm(image_interp, p=2, dim=1, keepdim=True).to(float64(image_interp)).add_(0.00001)

    # Interpolate the powered magnitudes, then un-power them (bring them back to a power of 1).
    a_magnitude = torch.norm(a, p=2, dim=1, keepdim=True).to(float64(a)).pow_(settings.inpaint_detail_preservation) * one_minus_t3
    b_magnitude = torch.norm(b, p=2, dim=1, keepdim=True).to(float64(b)).pow_(settings.inpaint_detail_preservation) * t3
    desired_magnitude = a_magnitude
    desired_magnitude.add_(b_magnitude).pow_(1 / settings.inpaint_detail_preservation)
    del a_magnitude, b_magnitude, t3, one_minus_t3

    # Change the linearly interpolated image vectors' magnitudes to the value we want.
    # This is the last 64-bit operation.
    image_interp_scaling_factor = desired_magnitude
    image_interp_scaling_factor.div_(current_magnitude)
    image_interp_scaling_factor = image_interp_scaling_factor.to(result_type)
    image_interp_scaled = image_interp
    image_interp_scaled.mul_(image_interp_scaling_factor)
    del current_magnitude
    del desired_magnitude
    del image_interp
    del image_interp_scaling_factor
    del result_type

    return image_interp_scaled


def get_modified_nmask(settings, nmask, sigma):
    """
    Converts a negative mask representing the transparency of the original latent vectors being overlaid
    to a mask that is scaled according to the denoising strength for this step.

    Where:
        0 = fully opaque, infinite density, fully masked
        1 = fully transparent, zero density, fully unmasked

    We bring this transparency to a power, as this allows one to simulate N number of blending operations
    where N can be any positive real value. Using this one can control the balance of influence between
    the denoiser and the original latents according to the sigma value.

    NOTE: "mask" is not used
    """
    import torch
    return torch.pow(nmask, (sigma ** settings.mask_blend_power) * settings.mask_blend_scale)


def apply_adaptive_masks(
        settings: SoftInpaintingSettings,
        nmask,
        latent_orig,
        latent_processed,
        overlay_images,
        width, height,
        paste_to):
    import torch
    import modules.processing as proc
    import modules.images as images
    from PIL import Image, ImageOps, ImageFilter

    # TODO: Bias the blending according to the latent mask, add adjustable parameter for bias control.
    if len(nmask.shape) == 3:
        latent_mask = nmask[0].float()
    else:
        latent_mask = nmask[:, 0].float()
    # convert the original mask into a form we use to scale distances for thresholding
    mask_scalar = 1 - (torch.clamp(latent_mask, min=0, max=1) ** (settings.mask_blend_scale / 2))
    mask_scalar = (0.5 * (1 - settings.composite_mask_influence)
                   + mask_scalar * settings.composite_mask_influence)
    mask_scalar = mask_scalar / (1.00001 - mask_scalar)
    mask_scalar = mask_scalar.cpu().numpy()

    latent_distance = torch.norm(latent_processed - latent_orig, p=2, dim=1)

    kernel, kernel_center = get_gaussian_kernel(stddev_radius=1.5, max_radius=2)

    masks_for_overlay = []

    for i, (distance_map, overlay_image) in enumerate(zip(latent_distance, overlay_images)):
        converted_mask = distance_map.float().cpu().numpy()
        converted_mask = weighted_histogram_filter(converted_mask, kernel, kernel_center,
                                                   percentile_min=0.9, percentile_max=1, min_width=1)
        converted_mask = weighted_histogram_filter(converted_mask, kernel, kernel_center,
                                                   percentile_min=0.25, percentile_max=0.75, min_width=1)

        # The distance at which opacity of original decreases to 50%
        if len(mask_scalar.shape) == 3:
            if mask_scalar.shape[0] > i:
                half_weighted_distance = settings.composite_difference_threshold * mask_scalar[i]
            else:
                half_weighted_distance = settings.composite_difference_threshold * mask_scalar[0]
        else:
            half_weighted_distance = settings.composite_difference_threshold * mask_scalar

        converted_mask = converted_mask / half_weighted_distance

        converted_mask = 1 / (1 + converted_mask ** settings.composite_difference_contrast)
        converted_mask = smootherstep(converted_mask)
        converted_mask = 1 - converted_mask
        converted_mask = 255. * converted_mask
        converted_mask = converted_mask.astype(np.uint8)
        converted_mask = Image.fromarray(converted_mask)
        converted_mask = images.resize_image(2, converted_mask, width, height)
        converted_mask = proc.create_binary_mask(converted_mask, round=False)

        # Remove aliasing artifacts using a gaussian blur.
        converted_mask = converted_mask.filter(ImageFilter.GaussianBlur(radius=4))

        # Expand the mask to fit the whole image if needed.
        if paste_to is not None:
            converted_mask = proc.uncrop(converted_mask,
                                         (overlay_image.width, overlay_image.height),
                                         paste_to)

        masks_for_overlay.append(converted_mask)

        image_masked = Image.new('RGBa', (overlay_image.width, overlay_image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"),
                           mask=ImageOps.invert(converted_mask.convert('L')))

        overlay_images[i] = image_masked.convert('RGBA')

    return masks_for_overlay


def apply_masks(
        settings,
        nmask,
        overlay_images,
        width, height,
        paste_to):
    import torch
    import modules.processing as proc
    import modules.images as images
    from PIL import Image, ImageOps, ImageFilter

    converted_mask = nmask[0].float()
    converted_mask = torch.clamp(converted_mask, min=0, max=1).pow_(settings.mask_blend_scale / 2)
    converted_mask = 255. * converted_mask
    converted_mask = converted_mask.cpu().numpy().astype(np.uint8)
    converted_mask = Image.fromarray(converted_mask)
    converted_mask = images.resize_image(2, converted_mask, width, height)
    converted_mask = proc.create_binary_mask(converted_mask, round=False)

    # Remove aliasing artifacts using a gaussian blur.
    converted_mask = converted_mask.filter(ImageFilter.GaussianBlur(radius=4))

    # Expand the mask to fit the whole image if needed.
    if paste_to is not None:
        converted_mask = proc.uncrop(converted_mask,
                                     (width, height),
                                     paste_to)

    masks_for_overlay = []

    for i, overlay_image in enumerate(overlay_images):
        masks_for_overlay[i] = converted_mask

        image_masked = Image.new('RGBa', (overlay_image.width, overlay_image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"),
                           mask=ImageOps.invert(converted_mask.convert('L')))

        overlay_images[i] = image_masked.convert('RGBA')

    return masks_for_overlay




def weighted_histogram_filter_single_pixel(idx, img, kernel, kernel_center, percentile_min, percentile_max, min_width):
    """
    Apply the weighted histogram filter to a single pixel.
    This function is now refactored to be accessible for parallelization.
    """
    idx = np.array(idx)
    kernel_min = -kernel_center
    kernel_max = np.array(kernel.shape) - kernel_center

    # Precompute the minimum and maximum valid indices for the kernel
    min_index = np.maximum(0, idx + kernel_min)
    max_index = np.minimum(np.array(img.shape), idx + kernel_max)
    window_shape = max_index - min_index

    # Initialize values and weights arrays
    values = []
    weights = []

    for window_tup in np.ndindex(*window_shape):
        window_index = np.array(window_tup)
        image_index = window_index + min_index
        centered_kernel_index = image_index - idx
        kernel_index = centered_kernel_index + kernel_center
        values.append(img[tuple(image_index)])
        weights.append(kernel[tuple(kernel_index)])

    # Convert to NumPy arrays
    values = np.array(values)
    weights = np.array(weights)

    # Sort values and weights by values
    sorted_indices = np.argsort(values)
    values = values[sorted_indices]
    weights = weights[sorted_indices]

    # Calculate cumulative weights
    cumulative_weights = np.cumsum(weights)

    # Define window boundaries
    sum_weights = cumulative_weights[-1]
    window_min = sum_weights * percentile_min
    window_max = sum_weights * percentile_max
    window_width = window_max - window_min

    # Ensure window is at least `min_width` wide
    if window_width < min_width:
        window_center = (window_min + window_max) / 2
        window_min = window_center - min_width / 2
        window_max = window_center + min_width / 2

        if window_max > sum_weights:
            window_max = sum_weights
            window_min = sum_weights - min_width

        if window_min < 0:
            window_min = 0
            window_max = min_width

    # Calculate overlap for each value
    overlap_start = np.maximum(window_min, np.concatenate(([0], cumulative_weights[:-1])))
    overlap_end = np.minimum(window_max, cumulative_weights)
    overlap = np.maximum(0, overlap_end - overlap_start)

    # Weighted average calculation
    result = np.sum(values * overlap) / np.sum(overlap) if np.sum(overlap) > 0 else 0
    return result

def weighted_histogram_filter(img, kernel, kernel_center, percentile_min=0.0, percentile_max=1.0, min_width=1.0, n_jobs=-1):
    """
    Generalization convolution filter capable of applying
    weighted mean, median, maximum, and minimum filters
    parametrically using an arbitrary kernel.

    Args:
        img (nparray):
            The image, a 2-D array of floats, to which the filter is being applied.
        kernel (nparray):
            The kernel, a 2-D array of floats.
        kernel_center (nparray):
            The kernel center coordinate, a 1-D array with two elements.
        percentile_min (float):
            The lower bound of the histogram window used by the filter,
            from 0 to 1.
        percentile_max (float):
            The upper bound of the histogram window used by the filter,
            from 0 to 1.
        min_width (float):
            The minimum size of the histogram window bounds, in weight units.
            Must be greater than 0.

    Returns:
        (nparray): A filtered copy of the input image "img", a 2-D array of floats.
    """

    # Ensure kernel_center is a 1D array
    if isinstance(kernel_center, int):
        kernel_center = np.array([kernel_center, kernel_center])
    elif len(kernel_center) == 1:
        kernel_center = np.array([kernel_center[0], kernel_center[0]])
    kernel_radius = max(kernel_center)
    padded_img = np.pad(img, kernel_radius, mode='constant', constant_values=0)
    img_out = np.zeros_like(img)
    img_shape = img.shape
    pixel_coords = [(i, j) for i in range(img_shape[0]) for j in range(img_shape[1])]

    def weighted_histogram_filter_single(idx):
        """
        Single-pixel weighted histogram calculation.
        """
        row, col = idx
        idx = (row + kernel_radius, col + kernel_radius)
        min_index = np.array(idx) - kernel_center
        max_index = min_index + kernel.shape

        window = padded_img[min_index[0]:max_index[0], min_index[1]:max_index[1]]
        window_values = window.flatten()
        window_weights = kernel.flatten()

        sorted_indices = np.argsort(window_values)
        values = window_values[sorted_indices]
        weights = window_weights[sorted_indices]

        cumulative_weights = np.cumsum(weights)
        sum_weights = cumulative_weights[-1]
        window_min = max(0, sum_weights * percentile_min)
        window_max = min(sum_weights, sum_weights * percentile_max)

        window_width = window_max - window_min
        if window_width < min_width:
            window_center = (window_min + window_max) / 2
            window_min = max(0, window_center - min_width / 2)
            window_max = min(sum_weights, window_center + min_width / 2)

        overlap_start = np.maximum(window_min, np.concatenate(([0], cumulative_weights[:-1])))
        overlap_end = np.minimum(window_max, cumulative_weights)
        overlap = np.maximum(0, overlap_end - overlap_start)

        return np.sum(values * overlap) / np.sum(overlap) if np.sum(overlap) > 0 else 0

    # Split pixel_coords into equal chunks based on n_jobs
    n_jobs = -1
    if cpu_count() > 6:
        n_jobs = 6 # More than 6 isn't worth unless it's more than 3000x3000px

    chunk_size = len(pixel_coords) // n_jobs
    pixel_chunks = [pixel_coords[i:i + chunk_size] for i in range(0, len(pixel_coords), chunk_size)]

    # joblib to process chunks in parallel
    def process_chunk(chunk):
        chunk_result = {}
        for idx in chunk:
            chunk_result[idx] = weighted_histogram_filter_single(idx)
        return chunk_result

    results = Parallel(n_jobs=n_jobs, backend="loky")( # loky is fastest in my configuration
        delayed(process_chunk)(chunk) for chunk in pixel_chunks
    )

    # Combine results into the output image
    for chunk_result in results:
        for (row, col), value in chunk_result.items():
            img_out[row, col] = value

    return img_out


def smoothstep(x):
    """
    The smoothstep function, input should be clamped to 0-1 range.
    Turns a diagonal line (f(x) = x) into a sigmoid-like curve.
    """
    return x * x * (3 - 2 * x)


def smootherstep(x):
    """
    The smootherstep function, input should be clamped to 0-1 range.
    Turns a diagonal line (f(x) = x) into a sigmoid-like curve.
    """
    return x * x * x * (x * (6 * x - 15) + 10)


def get_gaussian_kernel(stddev_radius=1.0, max_radius=2):
    """
    Creates a Gaussian kernel with thresholded edges.

    Args:
        stddev_radius (float):
            Standard deviation of the gaussian kernel, in pixels.
        max_radius (int):
            The size of the filter kernel. The number of pixels is (max_radius*2+1) ** 2.
            The kernel is thresholded so that any values one pixel beyond this radius
            is weighted at 0.

    Returns:
        (nparray, nparray): A kernel array (shape: (N, N)), its center coordinate (shape: (2))
    """

    # Evaluates a 0-1 normalized gaussian function for a given square distance from the mean.
    def gaussian(sqr_mag):
        return math.exp(-sqr_mag / (stddev_radius * stddev_radius))

    # Helper function for converting a tuple to an array.
    def vec(x):
        return np.array(x)

    """
    Since a gaussian is unbounded, we need to limit ourselves
    to a finite range.
    We taper the ends off at the end of that range so they equal zero
    while preserving the maximum value of 1 at the mean.
    """
    zero_radius = max_radius + 1.0
    gauss_zero = gaussian(zero_radius * zero_radius)
    gauss_kernel_scale = 1 / (1 - gauss_zero)

    def gaussian_kernel_func(coordinate):
        x = coordinate[0] ** 2.0 + coordinate[1] ** 2.0
        x = gaussian(x)
        x -= gauss_zero
        x *= gauss_kernel_scale
        x = max(0.0, x)
        return x

    size = max_radius * 2 + 1
    kernel_center = max_radius
    kernel = np.zeros((size, size))

    for index in np.ndindex(kernel.shape):
        kernel[index] = gaussian_kernel_func(vec(index) - kernel_center)

    return kernel, kernel_center


# ------------------- Constants -------------------


default = SoftInpaintingSettings(1, 0.5, 4, 0, 0.5, 2)

enabled_ui_label = "Soft inpainting"
enabled_gen_param_label = "Soft inpainting enabled"
enabled_el_id = "soft_inpainting_enabled"

ui_labels = SoftInpaintingSettings(
    "Schedule bias",
    "Preservation strength",
    "Transition contrast boost",
    "Mask influence",
    "Difference threshold",
    "Difference contrast")

ui_info = SoftInpaintingSettings(
    "Shifts when preservation of original content occurs during denoising.",
    "How strongly partially masked content should be preserved.",
    "Amplifies the contrast that may be lost in partially masked regions.",
    "How strongly the original mask should bias the difference threshold.",
    "How much an image region can change before the original pixels are not blended in anymore.",
    "How sharp the transition should be between blended and not blended.")

gen_param_labels = SoftInpaintingSettings(
    "Soft inpainting schedule bias",
    "Soft inpainting preservation strength",
    "Soft inpainting transition contrast boost",
    "Soft inpainting mask influence",
    "Soft inpainting difference threshold",
    "Soft inpainting difference contrast")

el_ids = SoftInpaintingSettings(
    "mask_blend_power",
    "mask_blend_scale",
    "inpaint_detail_preservation",
    "composite_mask_influence",
    "composite_difference_threshold",
    "composite_difference_contrast")


# ------------------- Script -------------------


class Script(scripts.Script):
    def __init__(self):
        self.section = "inpaint"
        self.masks_for_overlay = None
        self.overlay_images = None

    def title(self):
        return "Soft Inpainting"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        if not is_img2img:
            return

        with InputAccordion(False, label=enabled_ui_label, elem_id=enabled_el_id) as soft_inpainting_enabled:
            with gr.Group():
                gr.Markdown(
                    """
                    Soft inpainting allows you to **seamlessly blend original content with inpainted content** according to the mask opacity.
                    **High _Mask blur_** values are recommended!
                    """)

                power = \
                    gr.Slider(label=ui_labels.mask_blend_power,
                              info=ui_info.mask_blend_power,
                              minimum=0,
                              maximum=8,
                              step=0.1,
                              value=default.mask_blend_power,
                              elem_id=el_ids.mask_blend_power)
                scale = \
                    gr.Slider(label=ui_labels.mask_blend_scale,
                              info=ui_info.mask_blend_scale,
                              minimum=0,
                              maximum=8,
                              step=0.05,
                              value=default.mask_blend_scale,
                              elem_id=el_ids.mask_blend_scale)
                detail = \
                    gr.Slider(label=ui_labels.inpaint_detail_preservation,
                              info=ui_info.inpaint_detail_preservation,
                              minimum=1,
                              maximum=32,
                              step=0.5,
                              value=default.inpaint_detail_preservation,
                              elem_id=el_ids.inpaint_detail_preservation)

                gr.Markdown(
                    """
                    ### Pixel Composite Settings
                    """)

                mask_inf = \
                    gr.Slider(label=ui_labels.composite_mask_influence,
                              info=ui_info.composite_mask_influence,
                              minimum=0,
                              maximum=1,
                              step=0.05,
                              value=default.composite_mask_influence,
                              elem_id=el_ids.composite_mask_influence)

                dif_thresh = \
                    gr.Slider(label=ui_labels.composite_difference_threshold,
                              info=ui_info.composite_difference_threshold,
                              minimum=0,
                              maximum=8,
                              step=0.25,
                              value=default.composite_difference_threshold,
                              elem_id=el_ids.composite_difference_threshold)

                dif_contr = \
                    gr.Slider(label=ui_labels.composite_difference_contrast,
                              info=ui_info.composite_difference_contrast,
                              minimum=0,
                              maximum=8,
                              step=0.25,
                              value=default.composite_difference_contrast,
                              elem_id=el_ids.composite_difference_contrast)

                with gr.Accordion("Help", open=False):
                    gr.Markdown(
                        f"""
                        ### {ui_labels.mask_blend_power}

                        The blending strength of original content is scaled proportionally with the decreasing noise level values at each step (sigmas).
                        This ensures that the influence of the denoiser and original content preservation is roughly balanced at each step.
                        This balance can be shifted using this parameter, controlling whether earlier or later steps have stronger preservation.

                        - **Below 1**: Stronger preservation near the end (with low sigma)
                        - **1**: Balanced (proportional to sigma)
                        - **Above 1**: Stronger preservation in the beginning (with high sigma)
                        """)
                    gr.Markdown(
                        f"""
                        ### {ui_labels.mask_blend_scale}

                        Skews whether partially masked image regions should be more likely to preserve the original content or favor inpainted content.
                        This may need to be adjusted depending on the {ui_labels.mask_blend_power}, CFG Scale, prompt and Denoising strength.

                        - **Low values**: Favors generated content.
                        - **High values**: Favors original content.
                        """)
                    gr.Markdown(
                        f"""
                        ### {ui_labels.inpaint_detail_preservation}

                        This parameter controls how the original latent vectors and denoised latent vectors are interpolated.
                        With higher values, the magnitude of the resulting blended vector will be closer to the maximum of the two interpolated vectors.
                        This can prevent the loss of contrast that occurs with linear interpolation.

                        - **Low values**: Softer blending, details may fade.
                        - **High values**: Stronger contrast, may over-saturate colors.
                        """)

                    gr.Markdown(
                        """
                        ## Pixel Composite Settings

                        Masks are generated based on how much a part of the image changed after denoising.
                        These masks are used to blend the original and final images together.
                        If the difference is low, the original pixels are used instead of the pixels returned by the inpainting process.
                        """)

                    gr.Markdown(
                        f"""
                        ### {ui_labels.composite_mask_influence}

                        This parameter controls how much the mask should bias this sensitivity to difference.

                        - **0**: Ignore the mask, only consider differences in image content.
                        - **1**: Follow the mask closely despite image content changes.
                        """)

                    gr.Markdown(
                        f"""
                        ### {ui_labels.composite_difference_threshold}

                        This value represents the difference at which the original pixels will have less than 50% opacity.

                        - **Low values**: Two images patches must be almost the same in order to retain original pixels.
                        - **High values**: Two images patches can be very different and still retain original pixels.
                        """)

                    gr.Markdown(
                        f"""
                        ### {ui_labels.composite_difference_contrast}

                        This value represents the contrast between the opacity of the original and inpainted content.

                        - **Low values**: The blend will be more gradual and have longer transitions, but may cause ghosting.
                        - **High values**: Ghosting will be less common, but transitions may be very sudden.
                        """)

        self.infotext_fields = [(soft_inpainting_enabled, enabled_gen_param_label),
                                (power, gen_param_labels.mask_blend_power),
                                (scale, gen_param_labels.mask_blend_scale),
                                (detail, gen_param_labels.inpaint_detail_preservation),
                                (mask_inf, gen_param_labels.composite_mask_influence),
                                (dif_thresh, gen_param_labels.composite_difference_threshold),
                                (dif_contr, gen_param_labels.composite_difference_contrast)]

        self.paste_field_names = []
        for _, field_name in self.infotext_fields:
            self.paste_field_names.append(field_name)

        return [soft_inpainting_enabled,
                power,
                scale,
                detail,
                mask_inf,
                dif_thresh,
                dif_contr]

    def process(self, p, enabled, power, scale, detail_preservation, mask_inf, dif_thresh, dif_contr):
        if not enabled:
            return

        if not processing_uses_inpainting(p):
            return

        # Shut off the rounding it normally does.
        p.mask_round = False

        settings = SoftInpaintingSettings(power, scale, detail_preservation, mask_inf, dif_thresh, dif_contr)

        # p.extra_generation_params["Mask rounding"] = False
        settings.add_generation_params(p.extra_generation_params)

    def on_mask_blend(self, p, mba: scripts.MaskBlendArgs, enabled, power, scale, detail_preservation, mask_inf,
                      dif_thresh, dif_contr):
        if not enabled:
            return

        if not processing_uses_inpainting(p):
            return

        if mba.is_final_blend:
            mba.blended_latent = mba.current_latent
            return

        settings = SoftInpaintingSettings(power, scale, detail_preservation, mask_inf, dif_thresh, dif_contr)

        # todo: Why is sigma 2D? Both values are the same.
        mba.blended_latent = latent_blend(settings,
                                          mba.init_latent,
                                          mba.current_latent,
                                          get_modified_nmask(settings, mba.nmask, mba.sigma[0]))

    def post_sample(self, p, ps: scripts.PostSampleArgs, enabled, power, scale, detail_preservation, mask_inf,
                    dif_thresh, dif_contr):
        if not enabled:
            return

        if not processing_uses_inpainting(p):
            return

        nmask = getattr(p, "nmask", None)
        if nmask is None:
            return

        from modules import images
        from modules.shared import opts

        settings = SoftInpaintingSettings(power, scale, detail_preservation, mask_inf, dif_thresh, dif_contr)

        # since the original code puts holes in the existing overlay images,
        # we have to rebuild them.
        self.overlay_images = []
        for img in p.init_images:

            image = images.flatten(img, opts.img2img_background_color)

            if p.paste_to is None and p.resize_mode != 3:
                image = images.resize_image(p.resize_mode, image, p.width, p.height)

            self.overlay_images.append(image.convert('RGBA'))

        if len(p.init_images) == 1:
            self.overlay_images = self.overlay_images * p.batch_size

        if getattr(ps.samples, 'already_decoded', False):
            self.masks_for_overlay = apply_masks(settings=settings,
                                                 nmask=nmask,
                                                 overlay_images=self.overlay_images,
                                                 width=p.width,
                                                 height=p.height,
                                                 paste_to=p.paste_to)
        else:
            self.masks_for_overlay = apply_adaptive_masks(settings=settings,
                                                          nmask=nmask,
                                                          latent_orig=p.init_latent,
                                                          latent_processed=ps.samples,
                                                          overlay_images=self.overlay_images,
                                                          width=p.width,
                                                          height=p.height,
                                                          paste_to=p.paste_to)

    def postprocess_maskoverlay(self, p, ppmo: scripts.PostProcessMaskOverlayArgs, enabled, power, scale,
                                detail_preservation, mask_inf, dif_thresh, dif_contr):
        if not enabled:
            return

        if not processing_uses_inpainting(p):
            return

        if self.masks_for_overlay is None:
            return

        if self.overlay_images is None:
            return

        ppmo.mask_for_overlay = self.masks_for_overlay[ppmo.index]
        ppmo.overlay_image = self.overlay_images[ppmo.index]
