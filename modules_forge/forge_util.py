import torch
import numpy as np
import os
import time
import random
import string

from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn


def generate_random_filename(extension=".txt"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = f"{timestamp}-{random_string}{extension}"
    return filename


def cond_from_a1111_to_patched_ldm(cond):
    if isinstance(cond, torch.Tensor):
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=CONDCrossAttn(cond),
            )
        )
        return [result, ]

    cross_attn = cond['crossattn']
    pooled_output = cond['vector']

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=CONDCrossAttn(cross_attn),
            y=CONDRegular(pooled_output)
        )
    )

    return [result, ]


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
