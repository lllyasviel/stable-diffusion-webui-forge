from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict, TypedDict
import numpy as np
from modules import shared
from lib_controlnet.logging import logger
from lib_controlnet.enums import InputMode, HiResFixOption
from modules.api import api


def get_api_version() -> int:
    return 2


class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"


class BatchOption(Enum):
    DEFAULT = "All ControlNet units for all images in a batch"
    SEPARATE = "Each ControlNet unit for each image in a batch"


class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"


resize_mode_aliases = {
    'Inner Fit (Scale to Fit)': 'Crop and Resize',
    'Outer Fit (Shrink to Fit)': 'Resize and Fill',
    'Scale to Fit (Inner Fit)': 'Crop and Resize',
    'Envelope (Outer Fit)': 'Resize and Fill',
}


def resize_mode_from_value(value: Union[str, int, ResizeMode]) -> ResizeMode:
    if isinstance(value, str):
        return ResizeMode(resize_mode_aliases.get(value, value))
    elif isinstance(value, int):
        assert value >= 0
        if value == 3:  # 'Just Resize (Latent upscale)'
            return ResizeMode.RESIZE

        if value >= len(ResizeMode):
            logger.warning(f'Unrecognized ResizeMode int value {value}. Fall back to RESIZE.')
            return ResizeMode.RESIZE

        return [e for e in ResizeMode][value]
    else:
        return value


def control_mode_from_value(value: Union[str, int, ControlMode]) -> ControlMode:
    if isinstance(value, str):
        return ControlMode(value)
    elif isinstance(value, int):
        return [e for e in ControlMode][value]
    else:
        return value


def visualize_inpaint_mask(img):
    if img.ndim == 3 and img.shape[2] == 4:
        result = img.copy()
        mask = result[:, :, 3]
        mask = 255 - mask // 2
        result[:, :, 3] = mask
        return np.ascontiguousarray(result.copy())
    return img


def pixel_perfect_resolution(
        image: np.ndarray,
        target_H: int,
        target_W: int,
        resize_mode: ResizeMode,
) -> int:
    """
    Calculate the estimated resolution for resizing an image while preserving aspect ratio.

    The function first calculates scaling factors for height and width of the image based on the target
    height and width. Then, based on the chosen resize mode, it either takes the smaller or the larger
    scaling factor to estimate the new resolution.

    If the resize mode is OUTER_FIT, the function uses the smaller scaling factor, ensuring the whole image
    fits within the target dimensions, potentially leaving some empty space.

    If the resize mode is not OUTER_FIT, the function uses the larger scaling factor, ensuring the target
    dimensions are fully filled, potentially cropping the image.

    After calculating the estimated resolution, the function prints some debugging information.

    Args:
        image (np.ndarray): A 3D numpy array representing an image. The dimensions represent [height, width, channels].
        target_H (int): The target height for the image.
        target_W (int): The target width for the image.
        resize_mode (ResizeMode): The mode for resizing.

    Returns:
        int: The estimated resolution after resizing.
    """
    raw_H, raw_W, _ = image.shape

    k0 = float(target_H) / float(raw_H)
    k1 = float(target_W) / float(raw_W)

    if resize_mode == ResizeMode.OUTER_FIT:
        estimation = min(k0, k1) * float(min(raw_H, raw_W))
    else:
        estimation = max(k0, k1) * float(min(raw_H, raw_W))

    logger.debug(f"Pixel Perfect Computation:")
    logger.debug(f"resize_mode = {resize_mode}")
    logger.debug(f"raw_H = {raw_H}")
    logger.debug(f"raw_W = {raw_W}")
    logger.debug(f"target_H = {target_H}")
    logger.debug(f"target_W = {target_W}")
    logger.debug(f"estimation = {estimation}")

    return int(np.round(estimation))


class GradioImageMaskPair(TypedDict):
    """Represents the dict object from Gradio's image component if `tool="sketch"`
    is specified.
    {
        "image": np.ndarray,
        "mask": np.ndarray,
    }
    """
    image: np.ndarray
    mask: np.ndarray


@dataclass
class ControlNetUnit:
    """Represents an entire ControlNet processing unit.

    To add a new field to this class
    ## If the new field can be specified on UI, you need to
    - Add a new field of the same name in constructor of `ControlNetUiGroup`
    - Initialize the new `ControlNetUiGroup` field in `ControlNetUiGroup.render`
      as a Gradio `IOComponent`.
    - Add the new `ControlNetUiGroup` field to `unit_args` in
      `ControlNetUiGroup.render`. The order of parameters matters.

    ## If the new field needs to appear in infotext, you need to
    - Add a new item in `ControlNetUnit.infotext_fields`.
    API-only fields cannot appear in infotext.
    """
    # Following fields should only be used in the UI.
    # ====== Start of UI only fields ======
    # Specifies the input mode for the unit, defaulting to a simple mode.
    input_mode: InputMode = InputMode.SIMPLE
    # Determines whether to use the preview image as input; defaults to False.
    use_preview_as_input: bool = False
    # Directory path for batch processing of images.
    batch_image_dir: str = ''
    # Directory path for batch processing of masks.
    batch_mask_dir: str = ''
    # Optional list of gallery images for batch input; defaults to None.
    batch_input_gallery: Optional[List[str]] = None
    # Optional list of gallery masks for batch processing; defaults to None.
    batch_mask_gallery: Optional[List[str]] = None
    # Holds the preview image as a NumPy array; defaults to None.
    generated_image: Optional[np.ndarray] = None
    # ====== End of UI only fields ======

    # Following fields are used in both the API and the UI.
    # Holds the mask image; defaults to None.
    mask_image: Optional[GradioImageMaskPair] = None
    # Specifies how this unit should be applied in each pass of high-resolution fix.
    # Ignored if high-resolution fix is not enabled.
    hr_option: Union[HiResFixOption, int, str] = HiResFixOption.BOTH
    # Indicates whether the unit is enabled; defaults to True.
    enabled: bool = True
    # Name of the module being used; defaults to "None".
    module: str = "None"
    # Name of the model being used; defaults to "None".
    model: str = "None"
    # Weight of the unit in the overall processing; defaults to 1.0.
    weight: float = 1.0
    # Optional image for input; defaults to None.
    image: Optional[GradioImageMaskPair] = None
    # Specifies the mode of image resizing; defaults to inner fit.
    resize_mode: Union[ResizeMode, int, str] = ResizeMode.INNER_FIT
    # Resolution for processing by the unit; defaults to -1 (unspecified).
    processor_res: int = -1
    # Threshold A for processing; defaults to -1 (unspecified).
    threshold_a: float = -1
    # Threshold B for processing; defaults to -1 (unspecified).
    threshold_b: float = -1
    # Start value for guidance; defaults to 0.0.
    guidance_start: float = 0.0
    # End value for guidance; defaults to 1.0.
    guidance_end: float = 1.0
    # Enables pixel-perfect processing; defaults to False.
    pixel_perfect: bool = False
    # Control mode for the unit; defaults to balanced.
    control_mode: Union[ControlMode, int, str] = ControlMode.BALANCED

    # Following fields should only be used in the API.
    # ====== Start of API only fields ======
    # Whether to save the detected map for this unit; defaults to True.
    save_detected_map: bool = True
    # ====== End of API only fields ======

    @staticmethod
    def infotext_fields():
        """Fields that should be included in infotext.
        You should define a Gradio element with exact same name in ControlNetUiGroup
        as well, so that infotext can wire the value to correct field when pasting
        infotext.
        """
        return (
            "module",
            "model",
            "weight",
            "resize_mode",
            "processor_res",
            "threshold_a",
            "threshold_b",
            "guidance_start",
            "guidance_end",
            "pixel_perfect",
            "control_mode",
            "hr_option",
        )

    @staticmethod
    def from_dict(d: Dict) -> "ControlNetUnit":
        """Create ControlNetUnit from dict. This is primarily used to convert
        API json dict to ControlNetUnit."""
        unit = ControlNetUnit(
            **{k: v for k, v in d.items() if k in vars(ControlNetUnit)}
        )
        if isinstance(unit.image, str):
            img = np.array(api.decode_base64_to_image(unit.image)).astype('uint8')
            unit.image = {
                "image": img,
                "mask": np.zeros_like(img),
            }
        if isinstance(unit.mask_image, str):
            mask = np.array(api.decode_base64_to_image(unit.mask_image)).astype('uint8')
            if unit.image is not None:
                # Attach mask on image if ControlNet has input image.
                assert isinstance(unit.image, dict)
                unit.image["mask"] = mask
                unit.mask_image = None
            else:
                # Otherwise, wire to standalone mask.
                # This happens in img2img when using A1111 img2img input.
                unit.mask_image = {
                    "image": mask,
                    "mask": np.zeros_like(mask),
                }
        return unit


# Backward Compatible
UiControlNetUnit = ControlNetUnit


def to_base64_nparray(encoding: str):
    """
    Convert a base64 image into the image type the extension uses
    """

    return np.array(api.decode_base64_to_image(encoding)).astype('uint8')


def get_max_models_num():
    """
    Fetch the maximum number of allowed ControlNet models.
    """

    max_models_num = shared.opts.data.get("control_net_unit_count", 3)
    return max_models_num
