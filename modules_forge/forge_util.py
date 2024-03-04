import torch
import numpy as np
import os
import time
import random
import string
import cv2

from ldm_patched.modules import model_management


def prepare_free_memory(aggressive=False):
    if aggressive:
        model_management.unload_all_models()
        print('Cleanup all memory.')
        return

    model_management.free_memory(memory_required=model_management.minimum_inference_memory(),
                                 device=model_management.get_torch_device())
    print('Cleanup minimal inference memory.')
    return


def apply_circular_forge(model, tiling_enabled=False):
    if model.tiling_enabled == tiling_enabled:
        return

    print(f'Tiling: {tiling_enabled}')
    model.tiling_enabled = tiling_enabled

    def flatten(el):
        flattened = [flatten(children) for children in el.children()]
        res = [el]
        for c in flattened:
            res += c
        return res

    layers = flatten(model)

    for layer in [layer for layer in layers if 'Conv' in type(layer).__name__]:
        layer.padding_mode = 'circular' if tiling_enabled else 'zeros'
    return


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def generate_random_filename(extension=".txt"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = f"{timestamp}-{random_string}{extension}"
    return filename


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


def write_images_to_mp4(frame_list: list, filename=None, fps=6):
    from modules.paths_internal import default_output_dir

    video_folder = os.path.join(default_output_dir, 'svd')
    os.makedirs(video_folder, exist_ok=True)

    if filename is None:
        filename = generate_random_filename('.mp4')

    full_path = os.path.join(video_folder, filename)

    try:
        import av
    except ImportError:
        from launch import run_pip
        run_pip(
            "install imageio[pyav]",
            "imageio[pyav]",
        )
        import av

    options = {
        "crf": str(23)
    }

    output = av.open(full_path, "w")

    stream = output.add_stream('libx264', fps, options=options)
    stream.width = frame_list[0].shape[1]
    stream.height = frame_list[0].shape[0]
    for img in frame_list:
        frame = av.VideoFrame.from_ndarray(img)
        packet = stream.encode(frame)
        output.mux(packet)
    packet = stream.encode(None)
    output.mux(packet)
    output.close()

    return full_path


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(img, resolution):
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


def lazy_memory_management(model):
    required_memory = model_management.module_size(model) + model_management.minimum_inference_memory()
    model_management.free_memory(required_memory, device=model_management.get_torch_device())
    return
