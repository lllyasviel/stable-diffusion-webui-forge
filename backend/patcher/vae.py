import torch
import math
import itertools

from tqdm import trange
from backend import memory_management
from backend.patcher.base import ModelPatcher


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    dims = len(tile)
    output = torch.empty([samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])), device=output_device)

    for b in trange(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)

        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))
            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= ((1.0 / feather) * (t + 1))
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= ((1.0 / feather) * (t + 1))

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

        output[b:b + 1] = out / out_div
    return output


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device)


class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False):
        if no_init:
            return

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype)
        self.downscale_ratio = int(2 ** (len(model.config.down_block_types) - 1))
        self.latent_channels = int(model.config.latent_channels)

        self.first_stage_model = model.eval()

        if device is None:
            device = memory_management.vae_device()

        self.device = device
        offload_device = memory_management.vae_offload_device()

        if dtype is None:
            dtype = memory_management.vae_dtype()

        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = memory_management.intermediate_device()

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        return n

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()
        output = torch.clamp(((tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device))
                              / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        encode_fn = lambda a: self.first_stage_model.encode((2. * a - 1.).to(self.vae_dtype).to(self.device)).float()
        samples = tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples /= 3.0
        return samples

    def decode_inner(self, samples_in):
        if memory_management.VAE_ALWAYS_TILED:
            return self.decode_tiled(samples_in).to(self.output_device)

        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = memory_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * self.downscale_ratio), round(samples_in.shape[3] * self.downscale_ratio)), device=self.output_device)
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x:x + batch_number] = torch.clamp((self.first_stage_model.decode(samples).to(self.output_device).float() + 1.0) / 2.0, min=0.0, max=1.0)
        except memory_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            pixel_samples = self.decode_tiled_(samples_in)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode(self, samples_in):
        wrapper = self.patcher.model_options.get('model_vae_decode_wrapper', None)
        if wrapper is None:
            return self.decode_inner(samples_in)
        else:
            return wrapper(self.decode_inner, samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        memory_management.load_model_gpu(self.patcher)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        return output.movedim(1, -1)

    def encode_inner(self, pixel_samples):
        if memory_management.VAE_ALWAYS_TILED:
            return self.encode_tiled(pixel_samples)

        regulation = self.patcher.model_options.get("model_vae_regulation", None)

        pixel_samples = pixel_samples.movedim(-1, 1)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = memory_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)
            samples = torch.empty((pixel_samples.shape[0], self.latent_channels, round(pixel_samples.shape[2] // self.downscale_ratio), round(pixel_samples.shape[3] // self.downscale_ratio)), device=self.output_device)
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (2. * pixel_samples[x:x + batch_number] - 1.).to(self.vae_dtype).to(self.device)
                samples[x:x + batch_number] = self.first_stage_model.encode(pixels_in, regulation).to(self.output_device).float()

        except memory_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode(self, pixel_samples):
        wrapper = self.patcher.model_options.get('model_vae_encode_wrapper', None)
        if wrapper is None:
            return self.encode_inner(pixel_samples)
        else:
            return wrapper(self.encode_inner, pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        memory_management.load_model_gpu(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        return samples
