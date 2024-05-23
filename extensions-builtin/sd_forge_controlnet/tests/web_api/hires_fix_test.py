from sd_forge_kohya_hrfix.scripts.kohya_hrfix import KohyaHRFixForForge
from .template import (
    APITestTemplate,
    girl_img,
    disable_in_cq,
    get_model,
)

@disable_in_cq
def test_kohya_hrfix_with_sd15_controlnet():
    APITestTemplate(
        "test_kohya_hrfix_with_sd15_controlnet",
        "txt2img", 
        payload_overrides={
            "prompt": "a cat",
            "steps": 20,
            "width": 1024,
            "height": 1024,
        },
        unit_overrides={
            "image": girl_img,
            "module": "canny",
            "model": get_model("control_v11p_sd15_canny"),
        },
        script_args=(True, 3, 2.0, 0.0, 0.35, True, "bicubic", "bicubic"),
        scripts=[KohyaHRFixForForge()],
    ).exec()
