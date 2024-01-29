from modules_forge.controlnet import apply_controlnet_advanced


supported_control_model_types = []


class ControlModel:
    @staticmethod
    def try_build_from_state_dict(state_dict):
        return None, False

    def __init__(self, model_patcher):
        self.model_patcher = model_patcher

    def patch_to_process(self, p, control_image):
        return


class ControlNet(ControlModel):
    @staticmethod
    def try_build_from_state_dict(state_dict):
        return None, False

    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None

    def patch_to_process(self, p, control_image):
        unet = p.sd_model.forge_objects.unet

        # But in this simple example we do not use them
        positive_advanced_weighting = None
        negative_advanced_weighting = None
        advanced_frame_weighting = None
        advanced_sigma_weighting = None

        unet = apply_controlnet_advanced(
            unet=unet,
            controlnet=self.model_patcher,
            image_bchw=control_image,
            strength=0.6,
            start_percent=0.0,
            end_percent=0.8,
            positive_advanced_weighting=positive_advanced_weighting,
            negative_advanced_weighting=negative_advanced_weighting,
            advanced_frame_weighting=advanced_frame_weighting,
            advanced_sigma_weighting=advanced_sigma_weighting)

        p.sd_model.forge_objects.unet = unet
        return


supported_control_model_types.append(ControlNet)
