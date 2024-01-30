import os
from copy import copy
from typing import Dict, Optional, Tuple, List, Union
import modules.scripts as scripts
from modules import shared, script_callbacks, processing, masking, images
from modules.api.api import decode_base64_to_image
import gradio as gr

from lib_controlnet import global_state, external_code
from lib_controlnet.utils import align_dim_latent, image_dict_from_any, set_numpy_seed, crop_and_resize_image, prepare_mask
from lib_controlnet.enums import StableDiffusionVersion, HiResFixOption
from lib_controlnet.controlnet_ui.controlnet_ui_group import ControlNetUiGroup, UiControlNetUnit
from lib_controlnet.controlnet_ui.photopea import Photopea
from lib_controlnet.logging import logger
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing
from lib_controlnet.infotext import Infotext
from modules_forge.forge_util import HWC3, numpy_to_pytorch

import numpy as np
import functools

from PIL import Image
from modules_forge.shared import try_load_supported_control_model


# Gradio 3.32 bug fix
import tempfile
gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)


global_state.update_controlnet_filenames()


@functools.lru_cache(maxsize=shared.opts.data.get("control_net_model_cache_size", 5))
def cached_controlnet_loader(filename):
    return try_load_supported_control_model(filename)


class ControlNetCachedParameters:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.control_cond = None
        self.control_cond_for_hr_fix = None


class ControlNetForForgeOfficial(scripts.Script):
    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def get_default_ui_unit(is_ui=True):
        cls = UiControlNetUnit if is_ui else external_code.ControlNetUnit
        return cls(
            enabled=False,
            module="none",
            model="None"
        )

    def uigroup(self, tabname: str, is_img2img: bool, elem_id_tabname: str, photopea: Optional[Photopea]) -> Tuple[ControlNetUiGroup, gr.State]:
        group = ControlNetUiGroup(
            is_img2img,
            self.get_default_ui_unit(),
            photopea,
        )
        return group, group.render(tabname, elem_id_tabname)

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        infotext = Infotext()
        ui_groups = []
        controls = []
        max_models = shared.opts.data.get("control_net_unit_count", 3)
        elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_controlnet"
        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(f"ControlNet Integrated", open=False, elem_id="controlnet"):
                photopea = Photopea() if not shared.opts.data.get("controlnet_disable_photopea_edit", False) else None
                if max_models > 1:
                    with gr.Tabs(elem_id=f"{elem_id_tabname}_tabs"):
                        for i in range(max_models):
                            with gr.Tab(f"ControlNet Unit {i}",
                                        elem_classes=['cnet-unit-tab']):
                                group, state = self.uigroup(f"ControlNet-{i}", is_img2img, elem_id_tabname, photopea)
                                ui_groups.append(group)
                                controls.append(state)
                else:
                    with gr.Column():
                        group, state = self.uigroup(f"ControlNet", is_img2img, elem_id_tabname, photopea)
                        ui_groups.append(group)
                        controls.append(state)

        for i, ui_group in enumerate(ui_groups):
            infotext.register_unit(i, ui_group)
        if shared.opts.data.get("control_net_sync_field_args", True):
            self.infotext_fields = infotext.infotext_fields
            self.paste_field_names = infotext.paste_field_names

        return tuple(controls)

    @staticmethod
    def get_remote_call(p, attribute, default=None, idx=0, strict=False, force=False):
        if not force and not shared.opts.data.get("control_net_allow_script_control", False):
            return default

        def get_element(obj, strict=False):
            if not isinstance(obj, list):
                return obj if not strict or idx == 0 else None
            elif idx < len(obj):
                return obj[idx]
            else:
                return None

        attribute_value = get_element(getattr(p, attribute, None), strict)
        return attribute_value if attribute_value is not None else default

    def parse_remote_call(self, p, unit: external_code.ControlNetUnit, idx):
        selector = self.get_remote_call

        unit.enabled = selector(p, "control_net_enabled", unit.enabled, idx, strict=True)
        unit.module = selector(p, "control_net_module", unit.module, idx)
        unit.model = selector(p, "control_net_model", unit.model, idx)
        unit.weight = selector(p, "control_net_weight", unit.weight, idx)
        unit.image = selector(p, "control_net_image", unit.image, idx)
        unit.resize_mode = selector(p, "control_net_resize_mode", unit.resize_mode, idx)
        unit.low_vram = selector(p, "control_net_lowvram", unit.low_vram, idx)
        unit.processor_res = selector(p, "control_net_pres", unit.processor_res, idx)
        unit.threshold_a = selector(p, "control_net_pthr_a", unit.threshold_a, idx)
        unit.threshold_b = selector(p, "control_net_pthr_b", unit.threshold_b, idx)
        unit.guidance_start = selector(p, "control_net_guidance_start", unit.guidance_start, idx)
        unit.guidance_end = selector(p, "control_net_guidance_end", unit.guidance_end, idx)
        # Backward compatibility. See https://github.com/Mikubill/sd-webui-controlnet/issues/1740
        # for more details.
        unit.guidance_end = selector(p, "control_net_guidance_strength", unit.guidance_end, idx)
        unit.control_mode = selector(p, "control_net_control_mode", unit.control_mode, idx)
        unit.pixel_perfect = selector(p, "control_net_pixel_perfect", unit.pixel_perfect, idx)

        return unit

    def get_enabled_units(self, p):
        units = external_code.get_all_units_in_processing(p)
        if len(units) == 0:
            # fill a null group
            remote_unit = self.parse_remote_call(p, self.get_default_ui_unit(), 0)
            if remote_unit.enabled:
                units.append(remote_unit)

        enabled_units = []
        for idx, unit in enumerate(units):
            local_unit = self.parse_remote_call(p, unit, idx)
            if not local_unit.enabled:
                continue
            if hasattr(local_unit, "unfold_merged"):
                enabled_units.extend(local_unit.unfold_merged())
            else:
                enabled_units.append(copy(local_unit))

        Infotext.write_infotext(enabled_units, p)
        return enabled_units

    def choose_input_image(
            self,
            p: processing.StableDiffusionProcessing,
            unit: external_code.ControlNetUnit,
        ) -> Tuple[np.ndarray, external_code.ResizeMode]:
        """ Choose input image from following sources with descending priority:
         - p.image_control: [Deprecated] Lagacy way to pass image to controlnet.
         - p.control_net_input_image: [Deprecated] Lagacy way to pass image to controlnet.
         - unit.image: ControlNet tab input image.
         - p.init_images: A1111 img2img tab input image.

        Returns:
            - The input image in ndarray form.
            - The resize mode.
        """
        def parse_unit_image(unit: external_code.ControlNetUnit) -> Union[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
            unit_has_multiple_images = (
                isinstance(unit.image, list) and
                len(unit.image) > 0 and
                "image" in unit.image[0]
            )
            if unit_has_multiple_images:
                return [
                    d
                    for img in unit.image
                    for d in (image_dict_from_any(img),)
                    if d is not None
                ]
            return image_dict_from_any(unit.image)

        def decode_image(img) -> np.ndarray:
            """Need to check the image for API compatibility."""
            if isinstance(img, str):
                return np.asarray(decode_base64_to_image(image['image']))
            else:
                assert isinstance(img, np.ndarray)
                return img

        # 4 input image sources.
        image = parse_unit_image(unit)
        a1111_image = getattr(p, "init_images", [None])[0]

        resize_mode = external_code.resize_mode_from_value(unit.resize_mode)

        if image is not None:
            if isinstance(image, list):
                # Add mask logic if later there is a processor that accepts mask
                # on multiple inputs.
                input_image = [HWC3(decode_image(img['image'])) for img in image]
            else:
                input_image = HWC3(decode_image(image['image']))
                if 'mask' in image and image['mask'] is not None:
                    while len(image['mask'].shape) < 3:
                        image['mask'] = image['mask'][..., np.newaxis]
                    if 'inpaint' in unit.module:
                        logger.info("using inpaint as input")
                        color = HWC3(image['image'])
                        alpha = image['mask'][:, :, 0:1]
                        input_image = np.concatenate([color, alpha], axis=2)
                    elif (
                        not shared.opts.data.get("controlnet_ignore_noninpaint_mask", False) and
                        # There is wield gradio issue that would produce mask that is
                        # not pure color when no scribble is made on canvas.
                        # See https://github.com/Mikubill/sd-webui-controlnet/issues/1638.
                        not (
                            (image['mask'][:, :, 0] <= 5).all() or
                            (image['mask'][:, :, 0] >= 250).all()
                        )
                    ):
                        logger.info("using mask as input")
                        input_image = HWC3(image['mask'][:, :, 0])
                        unit.module = 'none'  # Always use black bg and white line
        elif a1111_image is not None:
            input_image = HWC3(np.asarray(a1111_image))
            a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
            assert a1111_i2i_resize_mode is not None
            resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

            a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
            if 'inpaint' in unit.module:
                if a1111_mask_image is not None:
                    a1111_mask = np.array(prepare_mask(a1111_mask_image, p))
                    assert a1111_mask.ndim == 2
                    assert a1111_mask.shape[0] == input_image.shape[0]
                    assert a1111_mask.shape[1] == input_image.shape[1]
                    input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                else:
                    input_image = np.concatenate([
                        input_image[:, :, 0:3],
                        np.zeros_like(input_image, dtype=np.uint8)[:, :, 0:1],
                    ], axis=2)
        else:
            raise ValueError("controlnet is enabled but no input image is given")

        assert isinstance(input_image, (np.ndarray, list))
        return input_image, resize_mode

    @staticmethod
    def try_crop_image_with_a1111_mask(
        p: StableDiffusionProcessing,
        unit: external_code.ControlNetUnit,
        input_image: np.ndarray,
        resize_mode: external_code.ResizeMode,
    ) -> np.ndarray:
        """
        Crop ControlNet input image based on A1111 inpaint mask given.
        This logic is crutial in upscale scripts, as they use A1111 mask + inpaint_full_res
        to crop tiles.
        """
        # Note: The method determining whether the active script is an upscale script is purely
        # based on `extra_generation_params` these scripts attach on `p`, and subject to change
        # in the future.
        # TODO: Change this to a more robust condition once A1111 offers a way to verify script name.
        is_upscale_script = any("upscale" in k.lower() for k in getattr(p, "extra_generation_params", {}).keys())
        logger.debug(f"is_upscale_script={is_upscale_script}")
        # Note: `inpaint_full_res` is "inpaint area" on UI. The flag is `True` when "Only masked"
        # option is selected.
        a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
        is_only_masked_inpaint = (
            issubclass(type(p), StableDiffusionProcessingImg2Img) and
            p.inpaint_full_res and
            a1111_mask_image is not None
        )
        if (
            'reference' not in unit.module
            and is_only_masked_inpaint
            and (is_upscale_script or unit.inpaint_crop_input_image)
        ):
            logger.debug("Crop input image based on A1111 mask.")
            input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
            input_image = [Image.fromarray(x) for x in input_image]

            mask = prepare_mask(a1111_mask_image, p)

            crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

            input_image = [
                images.resize_image(resize_mode.int_value(), i, mask.width, mask.height)
                for i in input_image
            ]

            input_image = [x.crop(crop_region) for x in input_image]
            input_image = [
                images.resize_image(external_code.ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height)
                for x in input_image
            ]

            input_image = [np.asarray(x)[:, :, 0] for x in input_image]
            input_image = np.stack(input_image, axis=2)
        return input_image

    @staticmethod
    def bound_check_params(unit: external_code.ControlNetUnit) -> None:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit'.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to
        their default values if negative.

        Args:
            unit (external_code.ControlNetUnit): The ControlNetUnit instance to check.
        """
        preprocessor = global_state.get_preprocessor(unit.module)

        if unit.processor_res < 0:
            unit.processor_res = int(preprocessor.slider_resolution.gradio_update_kwargs.get('value', 512))

        if unit.threshold_a < 0:
            unit.threshold_a = int(preprocessor.slider_1.gradio_update_kwargs.get('value', 1.0))

        if unit.threshold_b < 0:
            unit.threshold_b = int(preprocessor.slider_2.gradio_update_kwargs.get('value', 1.0))

        return

    @staticmethod
    def check_sd_version_compatible(unit: external_code.ControlNetUnit) -> None:
        """
        Checks whether the given ControlNet unit has model compatible with the currently
        active sd model. An exception is thrown if ControlNet unit is detected to be
        incompatible.
        """
        sd_version = global_state.get_sd_version()
        assert sd_version != StableDiffusionVersion.UNKNOWN

        if "revision" in unit.module.lower() and sd_version != StableDiffusionVersion.SDXL:
            raise Exception(f"Preprocessor 'revision' only supports SDXL. Current SD base model is {sd_version}.")

        # No need to check if the ControlModelType does not require model to be present.
        if unit.model is None or unit.model.lower() == "none":
            return

        cnet_sd_version = StableDiffusionVersion.detect_from_model_name(unit.model)

        if cnet_sd_version == StableDiffusionVersion.UNKNOWN:
            logger.warn(f"Unable to determine version for ControlNet model '{unit.model}'.")
            return

        if not sd_version.is_compatible_with(cnet_sd_version):
            raise Exception(f"ControlNet model {unit.model}({cnet_sd_version}) is not compatible with sd model({sd_version})")

    @staticmethod
    def get_target_dimensions(p: StableDiffusionProcessing) -> Tuple[int, int, int, int]:
        """Returns (h, w, hr_h, hr_w)."""
        h = align_dim_latent(p.height)
        w = align_dim_latent(p.width)

        high_res_fix = (
            isinstance(p, StableDiffusionProcessingTxt2Img)
            and getattr(p, 'enable_hr', False)
        )
        if high_res_fix:
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_y = int(p.height * p.hr_scale)
                hr_x = int(p.width * p.hr_scale)
            else:
                hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
            hr_y = align_dim_latent(hr_y)
            hr_x = align_dim_latent(hr_x)
        else:
            hr_y = h
            hr_x = w

        return h, w, hr_y, hr_x

    def process_unit_after_click_generate(self,
                                          p: StableDiffusionProcessing,
                                          unit: external_code.ControlNetUnit,
                                          params: ControlNetCachedParameters,
                                          *args, **kwargs):

        h, w, hr_y, hr_x = self.get_target_dimensions(p)

        has_high_res_fix = (
            isinstance(p, StableDiffusionProcessingTxt2Img)
            and getattr(p, 'enable_hr', False)
        )

        input_image, resize_mode = self.choose_input_image(p, unit)
        assert isinstance(input_image, np.ndarray), 'Invalid input image!'

        input_image = self.try_crop_image_with_a1111_mask(p, unit, input_image, resize_mode)
        input_image = np.ascontiguousarray(input_image.copy()).copy()  # safe numpy

        if unit.pixel_perfect:
            unit.processor_res = external_code.pixel_perfect_resolution(
                input_image,
                target_H=h,
                target_W=w,
                resize_mode=resize_mode,
            )

        seed = set_numpy_seed(p)
        logger.debug(f"Use numpy seed {seed}.")
        logger.info(f"Using preprocessor: {unit.module}")
        logger.info(f'preprocessor resolution = {unit.processor_res}')

        preprocessor = global_state.get_preprocessor(unit.module)

        detected_map = preprocessor(
            input_image=input_image,
            resolution=unit.processor_res,
            slider_1=unit.threshold_a,
            slider_2=unit.threshold_b,
        )

        detected_map_is_image = detected_map.ndim == 3 and detected_map.shape[2] < 5

        if detected_map_is_image:
            params.control_cond = crop_and_resize_image(detected_map, resize_mode, h, w)
            p.extra_result_images.append(params.control_cond)
            params.control_cond = numpy_to_pytorch(params.control_cond).movedim(-1, 1)

            if has_high_res_fix:
                params.control_cond_for_hr_fix = crop_and_resize_image(detected_map, resize_mode, hr_y, hr_x)
                p.extra_result_images.append(params.control_cond_for_hr_fix)
                params.control_cond_for_hr_fix = numpy_to_pytorch(params.control_cond_for_hr_fix).movedim(-1, 1)
            else:
                params.control_cond_for_hr_fix = params.control_cond
        else:
            params.control_cond = detected_map
            params.control_cond_for_hr_fix = detected_map
            p.extra_result_images.append(input_image)

        params.preprocessor = preprocessor

        model_filename = global_state.get_controlnet_filename(unit.model)
        params.model = cached_controlnet_loader(model_filename)

        logger.info(f"Current ControlNet: {model_filename}")

        return

    def process_unit_before_every_sampling(self,
                                           p: StableDiffusionProcessing,
                                           unit: external_code.ControlNetUnit,
                                           params: ControlNetCachedParameters,
                                           *args, **kwargs):

        if getattr(p, 'is_hr_pass', False):
            cond = params.control_cond_for_hr_fix
        else:
            cond = params.control_cond

        kwargs.update(dict(unit=unit, params=params))

        params.preprocessor.process_before_every_sampling(process=p, cond=cond, **kwargs)
        params.model.process_before_every_sampling(process=p, cond=cond, **kwargs)

        return

    def process(self, p, *args, **kwargs):
        self.current_params = {}
        for i, unit in enumerate(self.get_enabled_units(p)):
            self.bound_check_params(unit)
            params = ControlNetCachedParameters()
            self.process_unit_after_click_generate(p, unit, params, *args, **kwargs)
            self.current_params[i] = params
        return

    def process_before_every_sampling(self, p, *args, **kwargs):
        for unit, params in zip(self.get_enabled_units(p), self.current_params):
            self.process_unit_before_every_sampling(p, unit, params, *args, **kwargs)
        return

    def postprocess(self, p, processed, *args):
        return


def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        "detected_maps", "Directory for detected maps auto saving", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "", "Path to directory containing annotator model directories (requires restart, overrides corresponding command line flag)", section=section))
    shared.opts.add_option("control_net_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet: ControlNet unit number (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        5, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_inpaint_blur_sigma", shared.OptionInfo(
        7, "ControlNet inpainting Gaussian blur sigma", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=section))
    shared.opts.add_option("control_net_no_detectmap", shared.OptionInfo(
        False, "Do not append detectmap to output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_detectmap_autosaving", shared.OptionInfo(
        False, "Allow detectmap auto saving", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_allow_script_control", shared.OptionInfo(
        False, "Allow other script to control this extension", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("control_net_sync_field_args", shared.OptionInfo(
        True, "Paste ControlNet parameters in infotext", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_show_batch_images_in_ui", shared.OptionInfo(
        False, "Show batch images in gradio gallery output", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_increment_seed_during_batch", shared.OptionInfo(
        False, "Increment seed after each controlnet batch iteration", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_openpose_edit", shared.OptionInfo(
        False, "Disable openpose edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_photopea_edit", shared.OptionInfo(
        False, "Disable photopea edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_photopea_warning", shared.OptionInfo(
        True, "Photopea popup warning", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_ignore_noninpaint_mask", shared.OptionInfo(
        False, "Ignore mask on ControlNet input image if control type is not inpaint",
        gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_clip_detector_on_cpu", shared.OptionInfo(
        False, "Load CLIP preprocessor model on CPU",
        gr.Checkbox, {"interactive": True}, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
script_callbacks.on_after_component(ControlNetUiGroup.on_after_component)
script_callbacks.on_before_reload(ControlNetUiGroup.reset)