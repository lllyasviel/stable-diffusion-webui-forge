from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_ipadapter.IPAdapterPlus import IPAdapterApply

opIPAdapterApply = IPAdapterApply()


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

        if "ip_adapter" not in model.keys() or not model["ip_adapter"]:
            return None

        return IPAdapterPatcher(model)

    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self.ipadapter = model_patcher
        return

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        clip_vision, image = cond
        a = 0
        return


add_supported_control_model(IPAdapterPatcher)
