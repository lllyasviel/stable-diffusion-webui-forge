import torch


def apply_controlnet_advanced(
        unet,
        controlnet,
        image_bchw,
        strength,
        start_percent,
        end_percent,
        positive_advanced_weighting=None,
        negative_advanced_weighting=None,
        advanced_frame_weighting=None,
        advanced_sigma_weighting=None,
        advanced_mask_weighting=None
):
    """

    # positive_advanced_weighting or negative_advanced_weighting

    Unet has input, middle, output blocks, and we can give different weights to each layers in all blocks.
    Below is an example for stronger control in middle block.
    This is helpful for some high-res fix passes.

        positive_advanced_weighting = {
            'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'middle': [1.0],
            'output': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        }
        negative_advanced_weighting = {
            'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            'middle': [1.0],
            'output': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        }

    # advanced_frame_weighting

    The advanced_frame_weighting is a weight applied to each image in a batch.
    The length of this list must be same with batch size
    For example, if batch size is 5, you can use advanced_frame_weighting = [0, 0.25, 0.5, 0.75, 1.0]
    If you view the 5 images as 5 frames in a video, this will lead to progressively stronger control over time.

    # advanced_sigma_weighting

    The advanced_sigma_weighting allows you to dynamically compute control
    weights given diffusion timestep (sigma).
    For example below code can softly make beginning steps stronger than ending steps.

        sigma_max = unet.model.model_sampling.sigma_max
        sigma_min = unet.model.model_sampling.sigma_min
        advanced_sigma_weighting = lambda s: (s - sigma_min) / (sigma_max - sigma_min)

    # advanced_mask_weighting

    A mask can be applied to control signals.
    This should be a tensor with shape B 1 H W where the H and W can be arbitrary.
    This mask will be resized automatically to match the shape of all injection layers.

    """

    cnet = controlnet.copy().set_cond_hint(image_bchw, strength, (start_percent, end_percent))
    cnet.positive_advanced_weighting = positive_advanced_weighting
    cnet.negative_advanced_weighting = negative_advanced_weighting
    cnet.advanced_frame_weighting = advanced_frame_weighting
    cnet.advanced_sigma_weighting = advanced_sigma_weighting

    if advanced_mask_weighting is not None:
        assert isinstance(advanced_mask_weighting, torch.Tensor)
        B, C, H, W = advanced_mask_weighting.shape
        assert B > 0 and C == 1 and H > 0 and W > 0

    cnet.advanced_mask_weighting = advanced_mask_weighting

    m = unet.clone()
    m.add_patched_controlnet(cnet)
    return m

