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

    cnet = controlnet.copy().set_cond_hint(image_bhwc.movedim(-1, 1), strength, (start_percent, end_percent))
    cnet.positive_advanced_weighting = positive_advanced_weighting
    cnet.negative_advanced_weighting = negative_advanced_weighting
    cnet.advanced_frame_weighting = advanced_frame_weighting

    m = unet.clone()
    m.add_patched_controlnet(cnet)
    return m
