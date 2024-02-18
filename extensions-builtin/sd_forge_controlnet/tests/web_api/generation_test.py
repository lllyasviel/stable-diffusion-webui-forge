import pytest

from .template import (
    APITestTemplate,
    girl_img,
    mask_img,
    portrait_imgs,
    disable_in_cq,
    get_model,
)


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_no_unit(gen_type):
    assert APITestTemplate(
        f"test_no_unit{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides=[],
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_multiple_iter(gen_type):
    assert APITestTemplate(
        f"test_multiple_iter{gen_type}",
        gen_type,
        payload_overrides={"n_iter": 2},
        unit_overrides={},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_batch_size(gen_type):
    assert APITestTemplate(
        f"test_batch_size{gen_type}",
        gen_type,
        payload_overrides={"batch_size": 2},
        unit_overrides={},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_2_units(gen_type):
    assert APITestTemplate(
        f"test_2_units{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides=[{}, {}],
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_preprocessor(gen_type):
    assert APITestTemplate(
        f"test_preprocessor{gen_type}",
        gen_type,
        payload_overrides={},
        unit_overrides={"module": "canny"},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("param_name", ("processor_res", "threshold_a", "threshold_b"))
@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_invalid_param(gen_type, param_name):
    assert APITestTemplate(
        f"test_invalid_param{(gen_type, param_name)}",
        gen_type,
        payload_overrides={},
        unit_overrides={param_name: -1},
        input_image=girl_img,
    ).exec()


@pytest.mark.parametrize("save_map", [True, False])
@pytest.mark.parametrize("gen_type", ["img2img", "txt2img"])
def test_save_map(gen_type, save_map):
    assert APITestTemplate(
        f"test_save_map{(gen_type, save_map)}",
        gen_type,
        payload_overrides={},
        unit_overrides={"save_detected_map": save_map},
        input_image=girl_img,
    ).exec(expected_output_num=2 if save_map else 1)


@disable_in_cq
def test_masked_controlnet_txt2img():
    assert APITestTemplate(
        f"test_masked_controlnet_txt2img",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "image": girl_img,
            "mask_image": mask_img,
        },
    ).exec()


@disable_in_cq
def test_masked_controlnet_img2img():
    assert APITestTemplate(
        f"test_masked_controlnet_img2img",
        "img2img",
        payload_overrides={
            "init_images": [girl_img],
        },
        # Note: Currently you must give ControlNet unit input image to specify
        # mask.
        # TODO: Fix this for img2img.
        unit_overrides={
            "image": girl_img,
            "mask_image": mask_img,
        },
    ).exec()


@disable_in_cq
def test_txt2img_inpaint():
    assert APITestTemplate(
        "txt2img_inpaint",
        "txt2img",
        payload_overrides={},
        unit_overrides={
            "image": girl_img,
            "mask_image": mask_img,
            "model": get_model("v11p_sd15_inpaint"),
            "module": "inpaint_only",
        },
    ).exec()


@disable_in_cq
def test_img2img_inpaint():
    assert APITestTemplate(
        "img2img_inpaint",
        "img2img",
        payload_overrides={
            "init_images": [girl_img],
            "mask": mask_img,
        },
        unit_overrides={
            "model": get_model("v11p_sd15_inpaint"),
            "module": "inpaint_only",
        },
    ).exec()


# Currently failing.
# TODO Fix lama outpaint.
@disable_in_cq
def test_lama_outpaint():
    assert APITestTemplate(
        "txt2img_lama_outpaint",
        "txt2img",
        payload_overrides={
            "width": 768,
            "height": 768,
        },
        # Outpaint should not need a mask.
        unit_overrides={
            "image": girl_img,
            "model": get_model("v11p_sd15_inpaint"),
            "module": "inpaint_only+lama",
            "resize_mode": "Resize and Fill",  # OUTER_FIT
        },
    ).exec()


@disable_in_cq
def test_instant_id_sdxl():
    assert len(portrait_imgs) > 0
    assert APITestTemplate(
        "instant_id_sdxl",
        "txt2img",
        payload_overrides={
            "width": 1000,
            "height": 1000,
            "prompt": "1girl, red background",
        },
        unit_overrides=[
            dict(
                image=portrait_imgs[0],
                model=get_model("ip-adapter_instant_id_sdxl"),
                module="InsightFace (InstantID)",
            ),
            dict(
                image=portrait_imgs[1],
                model=get_model("control_instant_id_sdxl"),
                module="instant_id_face_keypoints",
            ),
        ],
    ).exec()


@disable_in_cq
def test_instant_id_sdxl_multiple_units():
    assert len(portrait_imgs) > 0
    assert APITestTemplate(
        "instant_id_sdxl_multiple_units",
        "txt2img",
        payload_overrides={
            "width": 1000,
            "height": 1000,
            "prompt": "1girl, red background",
        },
        unit_overrides=[
            dict(
                image=portrait_imgs[0],
                model=get_model("ip-adapter_instant_id_sdxl"),
                module="InsightFace (InstantID)",
            ),
            dict(
                image=portrait_imgs[1],
                model=get_model("control_instant_id_sdxl"),
                module="instant_id_face_keypoints",
            ),
            dict(
                image=portrait_imgs[1],
                model=get_model("diffusers_xl_canny"),
                module="canny",
            ),
        ],
    ).exec()
