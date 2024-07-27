# Forge Canvas
# AGPL V3
# by lllyasviel
# Commercial Use is not allowed. (Contact us for commercial use.)

import gradio.component_meta


create_or_modify_pyi_org = gradio.component_meta.create_or_modify_pyi


def create_or_modify_pyi_org_patched(component_class, class_name, events):
    try:
        if component_class.__name__ == 'LogicalImage':
            return
        return create_or_modify_pyi_org(component_class, class_name, events)
    except:
        return


gradio.component_meta.create_or_modify_pyi = create_or_modify_pyi_org_patched


import os
import uuid
import base64
import gradio as gr
import numpy as np

from PIL import Image
from io import BytesIO
from gradio.context import Context
from functools import wraps


canvas_js_root_path = os.path.dirname(__file__)


def web_js(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    return f'<script src="file={full_path}?{os.path.getmtime(full_path)}"></script>\n'


def web_css(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    return f'<link rel="stylesheet" href="file={full_path}?{os.path.getmtime(full_path)}">\n'


DEBUG_MODE = False

canvas_html = open(os.path.join(canvas_js_root_path, 'canvas.html'), encoding='utf-8').read()
canvas_head = ''
canvas_head += web_css('canvas.css')
canvas_head += web_js('canvas.min.js')


def image_to_base64(image_array, numpy=True):
    image = Image.fromarray(image_array) if numpy else image_array
    image = image.convert("RGBA")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


def base64_to_image(base64_str, numpy=True):
    if base64_str.startswith("data:image/png;base64,"):
        base64_str = base64_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGBA")
    image_array = np.array(image) if numpy else image
    return image_array


class LogicalImage(gr.Textbox):
    @wraps(gr.Textbox.__init__)
    def __init__(self, *args, numpy=True, **kwargs):
        self.numpy = numpy

        if 'value' in kwargs:
            initial_value = kwargs['value']
            if initial_value is not None:
                kwargs['value'] = self.image_to_base64(initial_value)
            else:
                del kwargs['value']

        super().__init__(*args, **kwargs)

    def preprocess(self, payload):
        if not isinstance(payload, str):
            return None

        if not payload.startswith("data:image/png;base64,"):
            return None

        return base64_to_image(payload, numpy=self.numpy)

    def postprocess(self, value):
        if value is None:
            return None

        return image_to_base64(value, numpy=self.numpy)

    def get_block_name(self):
        return "textbox"


class ForgeCanvas:
    def __init__(
            self,
            no_upload=False,
            no_scribbles=False,
            contrast_scribbles=False,
            height=512,
            scribble_color='#000000',
            scribble_color_fixed=False,
            scribble_width=4,
            scribble_width_fixed=False,
            scribble_alpha=100,
            scribble_alpha_fixed=False,
            scribble_softness=0,
            scribble_softness_fixed=False,
            visible=True,
            numpy=False,
            initial_image=None,
            elem_id=None,
            elem_classes=None
    ):
        self.uuid = 'uuid_' + uuid.uuid4().hex
        self.block = gr.HTML(canvas_html.replace('forge_mixin', self.uuid), visible=visible, elem_id=elem_id, elem_classes=elem_classes)
        self.foreground = LogicalImage(visible=DEBUG_MODE, label='foreground', numpy=numpy, elem_id=self.uuid, elem_classes=['logical_image_foreground'])
        self.background = LogicalImage(visible=DEBUG_MODE, label='background', numpy=numpy, value=initial_image, elem_id=self.uuid, elem_classes=['logical_image_background'])
        Context.root_block.load(None, js=f'async ()=>{{new ForgeCanvas("{self.uuid}", {no_upload}, {no_scribbles}, {contrast_scribbles}, {height}, '
                                         f"'{scribble_color}', {scribble_color_fixed}, {scribble_width}, {scribble_width_fixed}, "
                                         f'{scribble_alpha}, {scribble_alpha_fixed}, {scribble_softness}, {scribble_softness_fixed});}}')
