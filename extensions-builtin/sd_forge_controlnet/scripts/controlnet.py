import os
from typing import Dict, Optional, Tuple, List, Union

import cv2
import torch

import modules.scripts as scripts
from modules import shared, script_callbacks, masking, images
from modules.ui_components import InputAccordion
from modules.api.api import decode_base64_to_image
import gradio as gr

from lib_controlnet import global_state, external_code
from lib_controlnet.external_code import ControlNetUnit
from lib_controlnet.utils import align_dim_latent, set_numpy_seed, crop_and_resize_image, \
    prepare_mask, judge_image_type
from lib_controlnet.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
from lib_controlnet.controlnet_ui.photopea import Photopea
from lib_controlnet.logging import logger
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessing
from lib_controlnet.infotext import Infotext
from modules_forge.forge_util import HWC3, numpy_to_pytorch
from lib_controlnet.enums import HiResFixOption
from lib_controlnet.api import controlnet_api

import numpy as np
import functools

from PIL import Image
from modules_forge.shared import try_load_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher

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
        self.control_mask = None
        self.control_mask_for_hr_fix = None


class ControlNetForForgeOfficial(scripts.Script):
    sorting_priority = 10

    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        infotext = Infotext()
        ui_groups = []
        controls = []
        max_models = shared.opts.data.get("control_net_unit_count", 3)
        gen_type = "img2img" if is_img2img else "txt2img"
        elem_id_tabname = gen_type + "_controlnet"
        default_unit = ControlNetUnit(enabled=False, module="None", model="None")
        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(f"ControlNet Integrated", open=False, elem_id="controlnet",
                              elem_classes=["controlnet"]):
                photopea = (
                    Photopea()
                    if not shared.opts.data.get("controlnet_disable_photopea_edit", False)
                    else None
                )
                with gr.Row(elem_id=elem_id_tabname + "_accordions", elem_classes="accordions"):
                    for i in range(max_models):
                        with InputAccordion(
                            value=False,
                            label=f"ControlNet Unit {i}",
                            elem_classes=["cnet-unit-enabled-accordion"],  # Class on accordion
                        ):
                            group = ControlNetUiGroup(is_img2img, default_unit, photopea)
                            ui_groups.append(group)
                            controls.append(group.render(f"ControlNet-{i}", elem_id_tabname))

        for i, ui_group in enumerate(ui_groups):
            infotext.register_unit(i, ui_group)
        if shared.opts.data.get("control_net_sync_field_args", True):
            self.infotext_fields = infotext.infotext_fields
            self.paste_field_names = infotext.paste_field_names
        return tuple(controls)

    def get_enabled_units(self, units):
        # Parse dict from API calls.
        units = [
            ControlNetUnit.from_dict(unit) if isinstance(unit, dict) else unit
            for unit in units
        ]
        assert all(isinstance(unit, ControlNetUnit) for unit in units)
        enabled_units = [x for x in units if x.enabled]
        return enabled_units

    @staticmethod
    def try_crop_image_with_a1111_mask(
            p: StableDiffusionProcessing,
            unit: ControlNetUnit,
            input_image: np.ndarray,
            resize_mode: external_code.ResizeMode,
            preprocessor
    ) -> np.ndarray:
        a1111_mask_image: Optional[Image.Image] = getattr(p, "image_mask", None)
        is_only_masked_inpaint = (
                issubclass(type(p), StableDiffusionProcessingImg2Img) and
                p.inpaint_full_res and
                a1111_mask_image is not None
        )
        if (
                preprocessor.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab
                and is_only_masked_inpaint
        ):
            logger.info("Crop input image based on A1111 mask.")
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

    def get_input_data(self, p, unit, preprocessor, h, w):
        logger.info(f'ControlNet Input Mode: {unit.input_mode}')
        image_list = []
        resize_mode = external_code.resize_mode_from_value(unit.resize_mode)

        if unit.input_mode == external_code.InputMode.MERGE:
            for idx, item in enumerate(unit.batch_input_gallery):
                img_path = item['name']
                logger.info(f'Try to read image: {img_path}')
                img = np.ascontiguousarray(cv2.imread(img_path)[:, :, ::-1]).copy()
                mask = None
                if len(unit.batch_mask_gallery) > 0:
                    if len(unit.batch_mask_gallery) >= len(unit.batch_input_gallery):
                        mask_path = unit.batch_mask_gallery[idx]['name']
                    else:
                        mask_path = unit.batch_mask_gallery[0]['name']
                    mask = np.ascontiguousarray(cv2.imread(mask_path)[:, :, ::-1]).copy()
                if img is not None:
                    image_list.append([img, mask])
        elif unit.input_mode == external_code.InputMode.BATCH:
            image_list = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            batch_image_files = shared.listfiles(unit.batch_image_dir)
            for batch_modifier in getattr(unit, 'batch_modifiers', []):
                batch_image_files = batch_modifier(batch_image_files, p)
            for idx, filename in enumerate(batch_image_files):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    img_path = os.path.join(unit.batch_image_dir, filename)
                    logger.info(f'Try to read image: {img_path}')
                    img = np.ascontiguousarray(cv2.imread(img_path)[:, :, ::-1]).copy()
                    mask = None
                    if unit.batch_mask_dir:
                        batch_mask_files = shared.listfiles(unit.batch_mask_dir)
                        if len(batch_mask_files) >= len(batch_image_files):
                            mask_path = batch_mask_files[idx]
                        else:
                            mask_path = batch_mask_files[0]
                        mask_path = os.path.join(unit.batch_mask_dir, mask_path)
                        mask = np.ascontiguousarray(cv2.imread(mask_path)[:, :, ::-1]).copy()
                    if img is not None:
                        image_list.append([img, mask])
        else:
            a1111_i2i_image = getattr(p, "init_images", [None])[0]
            a1111_i2i_mask = getattr(p, "image_mask", None)

            using_a1111_data = False

            if unit.use_preview_as_input and unit.generated_image is not None:
                image = unit.generated_image
            elif unit.image is None:
                resize_mode = external_code.resize_mode_from_value(p.resize_mode)
                image = HWC3(np.asarray(a1111_i2i_image))
                using_a1111_data = True
            elif (unit.image['image'] < 5).all() and (unit.image['mask'] > 5).any():
                image = unit.image['mask']
            else:
                image = unit.image['image']

            if not isinstance(image, np.ndarray):
                raise ValueError("controlnet is enabled but no input image is given")

            image = HWC3(image)

            if using_a1111_data:
                mask = HWC3(np.asarray(a1111_i2i_mask)) if a1111_i2i_mask is not None else None
            elif unit.mask_image is not None and (unit.mask_image['image'] > 5).any():
                mask = unit.mask_image['image']
            elif unit.mask_image is not None and (unit.mask_image['mask'] > 5).any():
                mask = unit.mask_image['mask']
            elif unit.image is not None and (unit.image['mask'] > 5).any():
                mask = unit.image['mask']
            else:
                mask = None

            image = self.try_crop_image_with_a1111_mask(p, unit, image, resize_mode, preprocessor)

            if mask is not None:
                mask = cv2.resize(HWC3(mask), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = self.try_crop_image_with_a1111_mask(p, unit, mask, resize_mode, preprocessor)

            image_list = [[image, mask]]

        if resize_mode == external_code.ResizeMode.OUTER_FIT and preprocessor.expand_mask_when_resize_and_fill:
            new_image_list = []
            for input_image, input_mask in image_list:
                if input_mask is None:
                    input_mask = np.zeros_like(input_image)
                input_mask = crop_and_resize_image(
                    input_mask,
                    external_code.ResizeMode.OUTER_FIT, h, w,
                    fill_border_with_255=True,
                )
                input_image = crop_and_resize_image(
                    input_image,
                    external_code.ResizeMode.OUTER_FIT, h, w,
                    fill_border_with_255=False,
                )
                new_image_list.append((input_image, input_mask))
            image_list = new_image_list

        return image_list, resize_mode

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

    @torch.no_grad()
    def process_unit_after_click_generate(self,
                                          p: StableDiffusionProcessing,
                                          unit: ControlNetUnit,
                                          params: ControlNetCachedParameters,
                                          *args, **kwargs):

        h, w, hr_y, hr_x = self.get_target_dimensions(p)

        has_high_res_fix = (
                isinstance(p, StableDiffusionProcessingTxt2Img)
                and getattr(p, 'enable_hr', False)
        )

        if unit.use_preview_as_input:
            unit.module = 'None'

        preprocessor = global_state.get_preprocessor(unit.module)

        input_list, resize_mode = self.get_input_data(p, unit, preprocessor, h, w)
        preprocessor_outputs = []
        control_masks = []
        preprocessor_output_is_image = False
        preprocessor_output = None

        def optional_tqdm(iterable, use_tqdm):
            from tqdm import tqdm
            return tqdm(iterable) if use_tqdm else iterable

        for input_image, input_mask in optional_tqdm(input_list, len(input_list) > 1):
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

            preprocessor_output = preprocessor(
                input_image=input_image,
                input_mask=input_mask,
                resolution=unit.processor_res,
                slider_1=unit.threshold_a,
                slider_2=unit.threshold_b,
            )

            preprocessor_outputs.append(preprocessor_output)

            preprocessor_output_is_image = judge_image_type(preprocessor_output)

            if input_mask is not None:
                control_masks.append(input_mask)

            if len(input_list) > 1 and not preprocessor_output_is_image:
                logger.info('Batch wise input only support controlnet, control-lora, and t2i adapters!')
                break

        if has_high_res_fix:
            hr_option = HiResFixOption.from_value(unit.hr_option)
        else:
            hr_option = HiResFixOption.BOTH

        alignment_indices = [i % len(preprocessor_outputs) for i in range(p.batch_size)]
        def attach_extra_result_image(img: np.ndarray, is_high_res: bool = False):
            if (
                (is_high_res and hr_option.high_res_enabled) or
                (not is_high_res and hr_option.low_res_enabled)
            ) and unit.save_detected_map:
                p.extra_result_images.append(img)

        if preprocessor_output_is_image:
            params.control_cond = []
            params.control_cond_for_hr_fix = []

            for preprocessor_output in preprocessor_outputs:
                control_cond = crop_and_resize_image(preprocessor_output, resize_mode, h, w)
                attach_extra_result_image(external_code.visualize_inpaint_mask(control_cond))
                params.control_cond.append(numpy_to_pytorch(control_cond).movedim(-1, 1))

            params.control_cond = torch.cat(params.control_cond, dim=0)[alignment_indices].contiguous()

            if has_high_res_fix:
                for preprocessor_output in preprocessor_outputs:
                    control_cond_for_hr_fix = crop_and_resize_image(preprocessor_output, resize_mode, hr_y, hr_x)
                    attach_extra_result_image(external_code.visualize_inpaint_mask(control_cond_for_hr_fix), is_high_res=True)
                    params.control_cond_for_hr_fix.append(numpy_to_pytorch(control_cond_for_hr_fix).movedim(-1, 1))
                params.control_cond_for_hr_fix = torch.cat(params.control_cond_for_hr_fix, dim=0)[alignment_indices].contiguous()
            else:
                params.control_cond_for_hr_fix = params.control_cond
        else:
            params.control_cond = preprocessor_output
            params.control_cond_for_hr_fix = preprocessor_output
            attach_extra_result_image(input_image)

        if len(control_masks) > 0:
            params.control_mask = []
            params.control_mask_for_hr_fix = []

            for input_mask in control_masks:
                fill_border = preprocessor.fill_mask_with_one_when_resize_and_fill
                control_mask = crop_and_resize_image(input_mask, resize_mode, h, w, fill_border)
                attach_extra_result_image(control_mask)
                control_mask = numpy_to_pytorch(control_mask).movedim(-1, 1)[:, :1]
                params.control_mask.append(control_mask)

                if has_high_res_fix:
                    control_mask_for_hr_fix = crop_and_resize_image(input_mask, resize_mode, hr_y, hr_x, fill_border)
                    attach_extra_result_image(control_mask_for_hr_fix, is_high_res=True)
                    control_mask_for_hr_fix = numpy_to_pytorch(control_mask_for_hr_fix).movedim(-1, 1)[:, :1]
                    params.control_mask_for_hr_fix.append(control_mask_for_hr_fix)

            params.control_mask = torch.cat(params.control_mask, dim=0)[alignment_indices].contiguous()
            if has_high_res_fix:
                params.control_mask_for_hr_fix = torch.cat(params.control_mask_for_hr_fix, dim=0)[alignment_indices].contiguous()
            else:
                params.control_mask_for_hr_fix = params.control_mask

        if preprocessor.do_not_need_model:
            model_filename = 'Not Needed'
            params.model = ControlModelPatcher()
        else:
            assert unit.model != 'None', 'You have not selected any control model!'
            model_filename = global_state.get_controlnet_filename(unit.model)
            params.model = cached_controlnet_loader(model_filename)
            assert params.model is not None, logger.error(f"Recognizing Control Model failed: {model_filename}")

        params.preprocessor = preprocessor

        params.preprocessor.process_after_running_preprocessors(process=p, params=params, **kwargs)
        params.model.process_after_running_preprocessors(process=p, params=params, **kwargs)

        logger.info(f"Current ControlNet {type(params.model).__name__}: {model_filename}")
        return

    @torch.no_grad()
    def process_unit_before_every_sampling(self,
                                           p: StableDiffusionProcessing,
                                           unit: ControlNetUnit,
                                           params: ControlNetCachedParameters,
                                           *args, **kwargs):

        is_hr_pass = getattr(p, 'is_hr_pass', False)

        has_high_res_fix = (
                isinstance(p, StableDiffusionProcessingTxt2Img)
                and getattr(p, 'enable_hr', False)
        )

        if has_high_res_fix:
            hr_option = HiResFixOption.from_value(unit.hr_option)
        else:
            hr_option = HiResFixOption.BOTH

        if has_high_res_fix and is_hr_pass and (not hr_option.high_res_enabled):
            logger.info(f"ControlNet Skipped High-res pass.")
            return

        if has_high_res_fix and (not is_hr_pass) and (not hr_option.low_res_enabled):
            logger.info(f"ControlNet Skipped Low-res pass.")
            return

        if is_hr_pass:
            cond = params.control_cond_for_hr_fix
            mask = params.control_mask_for_hr_fix
        else:
            cond = params.control_cond
            mask = params.control_mask

        kwargs.update(dict(
            unit=unit,
            params=params,
            cond_original=cond.clone() if isinstance(cond, torch.Tensor) else cond,
            mask_original=mask.clone() if isinstance(mask, torch.Tensor) else mask,
        ))

        params.model.strength = float(unit.weight)
        params.model.start_percent = float(unit.guidance_start)
        params.model.end_percent = float(unit.guidance_end)
        params.model.positive_advanced_weighting = None
        params.model.negative_advanced_weighting = None
        params.model.advanced_frame_weighting = None
        params.model.advanced_sigma_weighting = None

        soft_weighting = {
            'input': [0.09941396206337118, 0.12050177219802567, 0.14606275417942507, 0.17704576264172736,
                      0.214600924414215,
                      0.26012233262329093, 0.3152997971191405, 0.3821815722656249, 0.4632503906249999, 0.561515625,
                      0.6806249999999999, 0.825],
            'middle': [0.561515625] if p.sd_model.is_sdxl else [1.0],
            'output': [0.09941396206337118, 0.12050177219802567, 0.14606275417942507, 0.17704576264172736,
                       0.214600924414215,
                       0.26012233262329093, 0.3152997971191405, 0.3821815722656249, 0.4632503906249999, 0.561515625,
                       0.6806249999999999, 0.825]
        }

        zero_weighting = {
            'input': [0.0] * 12,
            'middle': [0.0],
            'output': [0.0] * 12
        }

        if unit.control_mode == external_code.ControlMode.CONTROL.value:
            params.model.positive_advanced_weighting = soft_weighting.copy()
            params.model.negative_advanced_weighting = zero_weighting.copy()

        if unit.control_mode == external_code.ControlMode.PROMPT.value:
            params.model.positive_advanced_weighting = soft_weighting.copy()
            params.model.negative_advanced_weighting = soft_weighting.copy()

        if is_hr_pass and params.preprocessor.use_soft_projection_in_hr_fix:
            params.model.positive_advanced_weighting = soft_weighting.copy()
            params.model.negative_advanced_weighting = soft_weighting.copy()

        cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask, *args, **kwargs)

        params.model.advanced_mask_weighting = mask

        params.model.process_before_every_sampling(p, cond, mask, *args, **kwargs)

        logger.info(f"ControlNet Method {params.preprocessor.name} patched.")
        return

    @staticmethod
    def bound_check_params(unit: ControlNetUnit) -> None:
        """
        Checks and corrects negative parameters in ControlNetUnit 'unit'.
        Parameters 'processor_res', 'threshold_a', 'threshold_b' are reset to
        their default values if negative.

        Args:
            unit (ControlNetUnit): The ControlNetUnit instance to check.
        """
        preprocessor = global_state.get_preprocessor(unit.module)

        if unit.processor_res < 0:
            unit.processor_res = int(preprocessor.slider_resolution.gradio_update_kwargs.get('value', 512))

        if unit.threshold_a < 0:
            unit.threshold_a = int(preprocessor.slider_1.gradio_update_kwargs.get('value', 1.0))

        if unit.threshold_b < 0:
            unit.threshold_b = int(preprocessor.slider_2.gradio_update_kwargs.get('value', 1.0))

        return

    @torch.no_grad()
    def process_unit_after_every_sampling(self,
                                          p: StableDiffusionProcessing,
                                          unit: ControlNetUnit,
                                          params: ControlNetCachedParameters,
                                          *args, **kwargs):

        params.preprocessor.process_after_every_sampling(p, params, *args, **kwargs)
        params.model.process_after_every_sampling(p, params, *args, **kwargs)
        return

    @torch.no_grad()
    def process(self, p, *args, **kwargs):
        self.current_params = {}
        enabled_units = self.get_enabled_units(args)
        Infotext.write_infotext(enabled_units, p)
        for i, unit in enumerate(enabled_units):
            self.bound_check_params(unit)
            params = ControlNetCachedParameters()
            self.process_unit_after_click_generate(p, unit, params, *args, **kwargs)
            self.current_params[i] = params
        return

    @torch.no_grad()
    def process_before_every_sampling(self, p, *args, **kwargs):
        for i, unit in enumerate(self.get_enabled_units(args)):
            self.process_unit_before_every_sampling(p, unit, self.current_params[i], *args, **kwargs)
        return

    @torch.no_grad()
    def postprocess_batch_list(self, p, pp, *args, **kwargs):
        for i, unit in enumerate(self.get_enabled_units(args)):
            self.process_unit_after_every_sampling(p, unit, self.current_params[i], pp, *args, **kwargs)
        return

    def postprocess(self, p, processed, *args):
        self.current_params = {}
        return


def on_ui_settings():
    section = ('control_net', "ControlNet")
    shared.opts.add_option("control_net_detectedmap_dir", shared.OptionInfo(
        "detected_maps", "Directory for detected maps auto saving", section=section))
    shared.opts.add_option("control_net_models_path", shared.OptionInfo(
        "", "Extra path to scan for ControlNet models (e.g. training output directory)", section=section))
    shared.opts.add_option("control_net_modules_path", shared.OptionInfo(
        "",
        "Path to directory containing annotator model directories (requires restart, overrides corresponding command line flag)",
        section=section))
    shared.opts.add_option("control_net_unit_count", shared.OptionInfo(
        3, "Multi-ControlNet: ControlNet unit number (requires restart)", gr.Slider,
        {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("control_net_model_cache_size", shared.OptionInfo(
        5, "Model cache size (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
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
        False, "Increment seed after each controlnet batch iteration", gr.Checkbox, {"interactive": True},
        section=section))
    shared.opts.add_option("controlnet_disable_openpose_edit", shared.OptionInfo(
        False, "Disable openpose edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_disable_photopea_edit", shared.OptionInfo(
        False, "Disable photopea edit", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_photopea_warning", shared.OptionInfo(
        True, "Photopea popup warning", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("controlnet_input_thumbnail", shared.OptionInfo(
        True, "Input image thumbnail on unit header", gr.Checkbox, {"interactive": True}, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
script_callbacks.on_after_component(ControlNetUiGroup.on_after_component)
script_callbacks.on_before_reload(ControlNetUiGroup.reset)
script_callbacks.on_app_started(controlnet_api)
