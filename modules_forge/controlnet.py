def apply_controlnet_advanced(
        unet,
        controlnet,
        image_bhwc,
        strength,
        start_percent,
        end_percent,
        positive_advanced_weighting=None,
        negative_advanced_weighting=None,
        advanced_frame_weighting=None,
):
    """

    # positive_advanced_weighting or negative_advanced_weighting

    Unet has input, middle, output blocks, and we can give different weights to each layers in all blocks.
    Below is an example for stronger control in middle block.
    This is helpful for some high-res fix passes.

    positive_advanced_weighting = {
        'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        'middle': [1.0],
        'output': [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    }
    negative_advanced_weighting = {
        'input': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        'middle': [1.0],
        'output': [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    }

    # advanced_frame_weighting

    The advanced_frame_weighting is a weight applied to each image in a batch.
    The length of this list must be same with batch size
    For example, if batch size is 5, you can use advanced_frame_weighting = [0, 0.25, 0.5, 0.75, 1.0]
    If you view the 5 images as 5 frames in a video, this will lead to progressively stronger control over time.

    """

    cnet = controlnet.copy().set_cond_hint(image_bhwc.movedim(-1, 1), strength, (start_percent, end_percent))
    cnet.positive_advanced_weighting = positive_advanced_weighting
    cnet.negative_advanced_weighting = negative_advanced_weighting
    cnet.advanced_frame_weighting = advanced_frame_weighting

    m = unet.clone()
    m.add_patched_controlnet(cnet)
    return m
