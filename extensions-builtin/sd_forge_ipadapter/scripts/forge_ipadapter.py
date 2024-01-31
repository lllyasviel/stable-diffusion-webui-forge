from modules_forge.supported_preprocessor import PreprocessorClipVision, Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor
from modules_forge.forge_util import numpy_to_pytorch
from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_ipadapter.IPAdapterPlus import IPAdapterApply, InsightFaceLoader


opIPAdapterApply = IPAdapterApply().apply_ipadapter
opInsightFaceLoader = InsightFaceLoader().load_insight_face


class PreprocessorClipVisionForIPAdapter(PreprocessorClipVision):
    def __init__(self, name, url, filename):
        super().__init__(name, url, filename)
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']
        self.sorting_priority = 20

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            clip_vision=self.load_clipvision(),
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            attn_mask=None,
            unfold_batch=False,
        )
        return cond


add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-H (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors',
    filename='CLIP-ViT-H-14.safetensors'
))

add_supported_preprocessor(PreprocessorClipVisionForIPAdapter(
    name='CLIP-ViT-bigG (IPAdapter)',
    url='https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors',
    filename='CLIP-ViT-bigG.safetensors'
))


class PreprocessorInsightFaceForIPAdapter(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'InsightFace (IPAdapter)'
        self.tags = ['IP-Adapter']
        self.model_filename_filters = ['IP-Adapter', 'IP_Adapter']
        self.slider_resolution = PreprocessorParameter(visible=False)
        self.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = False
        self.show_control_mode = False
        self.sorting_priority = 1

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        cond = dict(
            insightface=opInsightFaceLoader("CPU")[0],
            image=numpy_to_pytorch(input_image),
            weight_type="original",
            noise=0.0,
            embeds=None,
            attn_mask=None,
            unfold_batch=False,
        )
        return cond


add_supported_preprocessor(PreprocessorInsightFaceForIPAdapter())


class IPAdapterPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        model = state_dict

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        if "ip_adapter" not in model.keys() or len(model["ip_adapter"]) == 0:
            return None

        return IPAdapterPatcher(model)

    def __init__(self, state_dict):
        super().__init__()
        self.ip_adapter = state_dict
        return

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = opIPAdapterApply(
            ipadapter=self.ip_adapter,
            model=unet,
            weight=self.strength,
            start_at=self.start_percent,
            end_at=self.end_percent,
            faceid_v2=False,
            weight_v2=False,
            **cond,
        )[0]

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(IPAdapterPatcher)
