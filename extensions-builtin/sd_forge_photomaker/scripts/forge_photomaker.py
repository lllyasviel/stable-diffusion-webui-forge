from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from ldm_patched.contrib.external_photomaker import PhotoMakerEncode, PhotoMakerIDEncoder


opPhotoMakerEncode = PhotoMakerEncode().apply_photomaker


class PreprocessorClipvisionForPhotomaker(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['PhotoMaker']
        self.model_filename_filters = ['PhotoMaker', 'Photo_Maker', 'Photo-Maker']
        self.sorting_priority = 20
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.show_control_mode = False


add_supported_preprocessor(PreprocessorClipvisionForPhotomaker(
    name='ClipVision (Photomaker)',
))


class PhotomakerPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if "id_encoder" not in state_dict:
            return None

        state_dict = state_dict["id_encoder"]

        photomaker_model = PhotoMakerIDEncoder()
        photomaker_model.load_state_dict(state_dict)

        return PhotomakerPatcher(photomaker_model)

    def __init__(self, model):
        super().__init__()
        self.model = model
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet.clone()
        clip = process.sd_model.forge_objects.clip
        text = process.prompts[0]

        cond_modified = opPhotoMakerEncode(photomaker=self.model, image=cond.movedim(1, -1), clip=clip, text=text)[0]
        cond_modified = unet.encode_conds_after_clip(conds=cond_modified, noise=kwargs['x'])[0]

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            cond = cond.copy()
            for c in cond:
                c['pooled_output'] = cond_modified['pooled_output']
                c['cross_attn'] = cond_modified['cross_attn']
                c['model_conds'].update(cond_modified['model_conds'])
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet.add_conditioning_modifier(conditioning_modifier)
        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(PhotomakerPatcher)
