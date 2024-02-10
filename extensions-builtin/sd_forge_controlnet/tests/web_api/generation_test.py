import pytest

from .template import (
    APITestTemplate,
    girl_img,
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
