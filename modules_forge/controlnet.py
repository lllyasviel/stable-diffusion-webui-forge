def apply_controlnet_advanced(
        unet,
        controlnet,
        cond_hint,
        strength,
        start_percent,
        end_percent,
        positive_advanced_weighting=None,
        negative_advanced_weighting=None):

    cnet = controlnet.copy().set_cond_hint(cond_hint, strength, (start_percent, end_percent))
    cnet.positive_advanced_weighting = positive_advanced_weighting
    cnet.negative_advanced_weighting = negative_advanced_weighting

    m = unet.clone()
    m.add_patched_controlnet(cnet)
    return m
