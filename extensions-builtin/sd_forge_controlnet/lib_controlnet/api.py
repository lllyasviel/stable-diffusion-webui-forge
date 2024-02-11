from typing import List

import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image
import gradio as gr

from modules.api import api
from .global_state import (
    get_all_preprocessor_names,
    get_all_controlnet_names,
    get_preprocessor,
)
from .utils import judge_image_type
from .logging import logger


def encode_to_base64(image):
    if isinstance(image, str):
        return image
    elif not judge_image_type(image):
        return "Detect result is not image"
    elif isinstance(image, Image.Image):
        return api.encode_pil_to_base64(image)
    elif isinstance(image, np.ndarray):
        return encode_np_to_base64(image)
    else:
        logger.warn("Unable to encode image.")
        return ""


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def controlnet_api(_: gr.Blocks, app: FastAPI):
    @app.get("/controlnet/model_list")
    async def model_list():
        up_to_date_model_list = get_all_controlnet_names()
        logger.debug(up_to_date_model_list)
        return {"model_list": up_to_date_model_list}

    @app.get("/controlnet/module_list")
    async def module_list():
        module_list = get_all_preprocessor_names()
        logger.debug(module_list)

        return {
            "module_list": module_list,
            # TODO: Add back module detail.
            # "module_detail": external_code.get_modules_detail(alias_names),
        }

    @app.post("/controlnet/detect")
    async def detect(
        controlnet_module: str = Body("none", title="Controlnet Module"),
        controlnet_input_images: List[str] = Body([], title="Controlnet Input Images"),
        controlnet_processor_res: int = Body(
            512, title="Controlnet Processor Resolution"
        ),
        controlnet_threshold_a: float = Body(64, title="Controlnet Threshold a"),
        controlnet_threshold_b: float = Body(64, title="Controlnet Threshold b"),
    ):
        processor_module = get_preprocessor(controlnet_module)
        if processor_module is None:
            raise HTTPException(status_code=422, detail="Module not available")

        if len(controlnet_input_images) == 0:
            raise HTTPException(status_code=422, detail="No image selected")

        logger.debug(
            f"Detecting {str(len(controlnet_input_images))} images with the {controlnet_module} module."
        )

        results = []
        poses = []

        for input_image in controlnet_input_images:
            img = np.array(api.decode_base64_to_image(input_image)).astype('uint8')

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = None

                def accept(self, json_dict: dict) -> None:
                    self.value = json_dict

            json_acceptor = JsonAcceptor()

            results.append(
                processor_module(
                    img,
                    resolution=controlnet_processor_res,
                    slider_1=controlnet_threshold_a,
                    slider_2=controlnet_threshold_b,
                    json_pose_callback=json_acceptor.accept,
                )
            )

            if "openpose" in controlnet_module:
                assert json_acceptor.value is not None
                poses.append(json_acceptor.value)

        results64 = [encode_to_base64(img) for img in results]
        res = {"images": results64, "info": "Success"}
        if poses:
            res["poses"] = poses

        return res

