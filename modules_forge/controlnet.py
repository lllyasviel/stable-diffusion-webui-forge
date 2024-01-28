def apply_controlnet_advanced(
        unet,
        controlnet,
        image,
        strength,
        start_percent,
        end_percent,
        positive_advanced_weighting=None,
        negative_advanced_weighting=None):

    a = 0

    unet.control_options = [1, 2, 3]

    m = unet.clone()

    m.control_options = [4, 5, 6]

    return m
