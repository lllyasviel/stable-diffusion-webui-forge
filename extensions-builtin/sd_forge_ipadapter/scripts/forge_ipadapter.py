from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher


class IPAdapterPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        a = 0
        return None

    def __init__(self, model_patcher):
        super().__init__(model_patcher)


add_supported_control_model(IPAdapterPatcher)
