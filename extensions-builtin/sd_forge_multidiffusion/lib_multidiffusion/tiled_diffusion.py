# Tiled Diffusion
# 1st edit by https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111
# 2nd edit by https://github.com/shiimizu/ComfyUI-TiledDiffusion
# 3rd edit by Forge Official


from __future__ import division
import torch
from torch import Tensor
import ldm_patched.modules.model_management
from ldm_patched.modules.model_patcher import ModelPatcher
import ldm_patched.modules.model_patcher
from ldm_patched.modules.model_base import BaseModel
from typing import List, Union, Tuple, Dict
from ldm_patched.contrib.external import ImageScale
import ldm_patched.modules.utils
from ldm_patched.modules.controlnet import ControlNet, T2IAdapter

opt_C = 4
opt_f = 8

def ceildiv(big, small):
    # Correct ceiling division that avoids floating-point errors and importing math.ceil.
    return -(big // -small)

from enum import Enum
class BlendMode(Enum):  # i.e. LayerType
    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'

class Processing: ...
class Device: ...
devices = Device()
devices.device = ldm_patched.modules.model_management.get_torch_device()

def null_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

keep_signature = null_decorator
controlnet     = null_decorator
stablesr       = null_decorator
grid_bbox      = null_decorator
custom_bbox    = null_decorator
noise_inverse  = null_decorator

class BBox:
    ''' grid bbox '''

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]

def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight

class CustomBBox(BBox):
    ''' region control bbox '''
    pass

class AbstractDiffusion:
    def __init__(self):
        self.method = self.__class__.__name__
        self.pbar = None


        self.w: int = 0
        self.h: int = 0
        self.tile_width: int = None
        self.tile_height: int = None
        self.tile_overlap: int = None
        self.tile_batch_size: int = None

        # cache. final result of current sampling step, [B, C=4, H//8, W//8]
        # avoiding overhead of creating new tensors and weight summing
        self.x_buffer: Tensor = None
        # self.w: int = int(self.p.width  // opt_f)       # latent size
        # self.h: int = int(self.p.height // opt_f)
        # weights for background & grid bboxes
        self._weights: Tensor = None
        # self.weights: Tensor = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self._init_grid_bbox = None
        self._init_done = None

        # count the step correctly
        self.step_count = 0         
        self.inner_loop_count = 0  
        self.kdiff_step = -1

        # ext. Grid tiling painting (grid bbox)
        self.enable_grid_bbox: bool = False
        self.tile_w: int = None
        self.tile_h: int = None
        self.tile_bs: int = None
        self.num_tiles: int = None
        self.num_batches: int = None
        self.batched_bboxes: List[List[BBox]] = []

        # ext. Region Prompt Control (custom bbox)
        self.enable_custom_bbox: bool = False
        self.custom_bboxes: List[CustomBBox] = []
        # self.cond_basis: Cond = None
        # self.uncond_basis: Uncond = None
        # self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        # self.causal_layers: bool = None

        # ext. ControlNet
        self.enable_controlnet: bool = False
        # self.controlnet_script: ModuleType = None
        self.control_tensor_batch_dict = {}
        self.control_tensor_batch: List[List[Tensor]] = [[]]
        # self.control_params: Dict[str, Tensor] = None # {}
        self.control_params: Dict[Tuple, List[List[Tensor]]] = {}
        self.control_tensor_cpu: bool = None
        self.control_tensor_custom: List[List[Tensor]] = []

        self.draw_background: bool = True       # by default we draw major prompts in grid tiles
        self.control_tensor_cpu = False
        self.weights = None
        self.imagescale = ImageScale()

    def reset(self):
        tile_width = self.tile_width
        tile_height = self.tile_height
        tile_overlap = self.tile_overlap
        tile_batch_size = self.tile_batch_size
        self.__init__()
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size

    def repeat_tensor(self, x:Tensor, n:int, concat=False, concat_to=0) -> Tensor:
        ''' repeat the tensor on it's first dim '''
        if n == 1: return x
        B = x.shape[0]
        r_dims = len(x.shape) - 1
        if B == 1:      # batch_size = 1 (not `tile_batch_size`)
            shape = [n] + [-1] * r_dims     # [N, -1, ...]
            return x.expand(shape)          # `expand` is much lighter than `tile`
        else:
            if concat:
                return torch.cat([x for _ in range(n)], dim=0)[:concat_to]
            shape = [n] + [1] * r_dims      # [N, 1, ...]
            return x.repeat(shape)
    def update_pbar(self):
        if self.pbar.n >= self.pbar.total:
            self.pbar.close()
        else:
            # self.pbar.update()
            sampling_step = 20
            if self.step_count == sampling_step:
                self.inner_loop_count += 1
                if self.inner_loop_count < self.total_bboxes:
                    self.pbar.update()
            else:
                self.step_count = sampling_step
                self.inner_loop_count = 0
    def reset_buffer(self, x_in:Tensor):
        # Judge if the shape of x_in is the same as the shape of x_buffer
        if self.x_buffer is None or self.x_buffer.shape != x_in.shape:
            self.x_buffer = torch.zeros_like(x_in, device=x_in.device, dtype=x_in.dtype)
        else:
            self.x_buffer.zero_()

    @grid_bbox
    def init_grid_bbox(self, tile_w:int, tile_h:int, overlap:int, tile_bs:int):
        # if self._init_grid_bbox is not None: return
        # self._init_grid_bbox = True
        self.weights = torch.zeros((1, 1, self.h, self.w), device=devices.device, dtype=torch.float32)
        self.enable_grid_bbox = True

        self.tile_w = min(tile_w, self.w)
        self.tile_h = min(tile_h, self.h)
        overlap = max(0, min(overlap, min(tile_w, tile_h) - 4))
        # split the latent into overlapped tiles, then batching
        # weights basically indicate how many times a pixel is painted
        bboxes, weights = split_bboxes(self.w, self.h, self.tile_w, self.tile_h, overlap, self.get_tile_weights())
        self.weights += weights
        self.num_tiles = len(bboxes)
        self.num_batches = ceildiv(self.num_tiles , tile_bs)
        self.tile_bs = ceildiv(len(bboxes) , self.num_batches)          # optimal_batch_size
        self.batched_bboxes = [bboxes[i*self.tile_bs:(i+1)*self.tile_bs] for i in range(self.num_batches)]

    @grid_bbox
    def get_tile_weights(self) -> Union[Tensor, float]:
        return 1.0

    @noise_inverse
    def init_noise_inverse(self, steps:int, retouch:float, get_cache_callback, set_cache_callback, renoise_strength:float, renoise_kernel:int):
        self.noise_inverse_enabled = True
        self.noise_inverse_steps = steps
        self.noise_inverse_retouch = float(retouch)
        self.noise_inverse_renoise_strength = float(renoise_strength)
        self.noise_inverse_renoise_kernel = int(renoise_kernel)
        self.noise_inverse_set_cache = set_cache_callback
        self.noise_inverse_get_cache = get_cache_callback

    def init_done(self):
        '''
          Call this after all `init_*`, settings are done, now perform:
            - settings sanity check 
            - pre-computations, cache init
            - anything thing needed before denoising starts
        '''

        # if self._init_done is not None: return
        # self._init_done = True
        self.total_bboxes = 0
        if self.enable_grid_bbox:   self.total_bboxes += self.num_batches
        if self.enable_custom_bbox: self.total_bboxes += len(self.custom_bboxes)
        assert self.total_bboxes > 0, "Nothing to paint! No background to draw and no custom bboxes were provided."

        # sampling_steps = _steps
        # self.pbar = tqdm(total=(self.total_bboxes) * sampling_steps, desc=f"{self.method} Sampling: ")

    @controlnet
    def prepare_controlnet_tensors(self, refresh:bool=False, tensor=None):
        ''' Crop the control tensor into tiles and cache them '''
        if not refresh:
            if self.control_tensor_batch is not None or self.control_params is not None: return
        tensors = [tensor]
        self.org_control_tensor_batch = tensors
        self.control_tensor_batch = []
        for i in range(len(tensors)):
            control_tile_list = []
            control_tensor = tensors[i]
            for bboxes in self.batched_bboxes:
                single_batch_tensors = []
                for bbox in bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    single_batch_tensors.append(control_tile)
                control_tile = torch.cat(single_batch_tensors, dim=0)
                if self.control_tensor_cpu:
                    control_tile = control_tile.cpu()
                control_tile_list.append(control_tile)
            self.control_tensor_batch.append(control_tile_list)

            if len(self.custom_bboxes) > 0:
                custom_control_tile_list = []
                for bbox in self.custom_bboxes:
                    if len(control_tensor.shape) == 3:
                        control_tensor.unsqueeze_(0)
                    control_tile = control_tensor[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f]
                    if self.control_tensor_cpu:
                        control_tile = control_tile.cpu()
                    custom_control_tile_list.append(control_tile)
                self.control_tensor_custom.append(custom_control_tile_list)

    @controlnet
    def switch_controlnet_tensors(self, batch_id:int, x_batch_size:int, tile_batch_size:int, is_denoise=False):
        # if not self.enable_controlnet: return
        if self.control_tensor_batch is None: return
        # self.control_params = [0]

        # for param_id in range(len(self.control_params)):
        for param_id in range(len(self.control_tensor_batch)):
            # tensor that was concatenated in `prepare_controlnet_tensors`
            control_tile = self.control_tensor_batch[param_id][batch_id]
            # broadcast to latent batch size
            if x_batch_size > 1: # self.is_kdiff:
                all_control_tile = []
                for i in range(tile_batch_size):
                    this_control_tile = [control_tile[i].unsqueeze(0)] * x_batch_size
                    all_control_tile.append(torch.cat(this_control_tile, dim=0))
                control_tile = torch.cat(all_control_tile, dim=0) # [:x_tile.shape[0]]
                self.control_tensor_batch[param_id][batch_id] = control_tile
            # else:
            #     control_tile = control_tile.repeat([x_batch_size if is_denoise else x_batch_size * 2, 1, 1, 1])
            # self.control_params[param_id].hint_cond = control_tile.to(devices.device)

    def process_controlnet(self, x_shape, x_dtype, c_in: dict, cond_or_uncond: List, bboxes, batch_size: int, batch_id: int):
        control: ControlNet = c_in['control_model']
        param_id = -1 # current controlnet & previous_controlnets
        tuple_key = tuple(cond_or_uncond) + tuple(x_shape)
        while control is not None:
            param_id += 1
            PH, PW = self.h*8, self.w*8
            
            if self.control_params.get(tuple_key, None) is None:
                self.control_params[tuple_key] = [[None]]
                val = self.control_params[tuple_key]
                if param_id+1 >= len(val):
                    val.extend([[None] for _ in range(param_id+1)])
                if len(self.batched_bboxes) >= len(val[param_id]):
                    val[param_id].extend([[None] for _ in range(len(self.batched_bboxes))])

            # Below is taken from ldm_patched.modules.controlnet.py, but we need to additionally tile the cnets.
            # if statement: eager eval. first time when cond_hint is None. 
            if self.refresh or control.cond_hint is None or not isinstance(self.control_params[tuple_key][param_id][batch_id], Tensor):
                dtype = getattr(control, 'manual_cast_dtype', None)
                if dtype is None: dtype = getattr(getattr(control, 'control_model', None), 'dtype', None)
                if dtype is None: dtype = x_dtype
                if isinstance(control, T2IAdapter):
                    width, height = control.scale_image_to(PW, PH)
                    control.cond_hint = ldm_patched.modules.utils.common_upscale(control.cond_hint_original, width, height, 'nearest-exact', "center").float().to(control.device)
                    if control.channels_in == 1 and control.cond_hint.shape[1] > 1:
                        control.cond_hint = torch.mean(control.cond_hint, 1, keepdim=True)
                elif control.__class__.__name__ == 'ControlLLLiteAdvanced':
                    if control.sub_idxs is not None and control.cond_hint_original.shape[0] >= control.full_latent_length:
                        control.cond_hint = ldm_patched.modules.utils.common_upscale(control.cond_hint_original[control.sub_idxs], PW, PH, 'nearest-exact', "center").to(dtype=dtype, device=control.device)
                    else:
                        if (PH, PW) == (control.cond_hint_original.shape[-2], control.cond_hint_original.shape[-1]):
                            control.cond_hint = control.cond_hint_original.clone().to(dtype=dtype, device=control.device)
                        else:
                            control.cond_hint = ldm_patched.modules.utils.common_upscale(control.cond_hint_original, PW, PH, 'nearest-exact', "center").to(dtype=dtype, device=control.device)
                else:
                    if (PH, PW) == (control.cond_hint_original.shape[-2], control.cond_hint_original.shape[-1]):
                        control.cond_hint = control.cond_hint_original.clone().to(dtype=dtype, device=control.device)
                    else:
                        control.cond_hint = ldm_patched.modules.utils.common_upscale(control.cond_hint_original, PW, PH, 'nearest-exact', 'center').to(dtype=dtype, device=control.device)
                
                # Broadcast then tile
                #
                # Below can be in the parent's if clause because self.refresh will trigger on resolution change, e.g. cause of ConditioningSetArea
                # so that particular case isn't cached atm.
                cond_hint_pre_tile = control.cond_hint
                if control.cond_hint.shape[0] < batch_size :
                    cond_hint_pre_tile = self.repeat_tensor(control.cond_hint, ceildiv(batch_size, control.cond_hint.shape[0]))[:batch_size]
                cns = [cond_hint_pre_tile[:, :, bbox[1]*opt_f:bbox[3]*opt_f, bbox[0]*opt_f:bbox[2]*opt_f] for bbox in bboxes]
                control.cond_hint = torch.cat(cns, dim=0)
                self.control_params[tuple_key][param_id][batch_id]=control.cond_hint
            else:
                control.cond_hint = self.control_params[tuple_key][param_id][batch_id]
            control = control.previous_controlnet

import numpy as np
from numpy import pi, exp, sqrt
def gaussian_weights(tile_w:int, tile_h:int) -> Tensor:
    '''
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.
    '''
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]   # -1 because index goes from 0 to latent_width - 1
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)

class CondDict: ...

class MultiDiffusion(AbstractDiffusion):
    
    @torch.no_grad()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]
        c_crossattn: Tensor = c_in['c_crossattn']

        N, C, H, W = x_in.shape

        # ldm_patched.modulesui can feed in a latent that's a different size cause of SetArea, so we'll refresh in that case.
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if ldm_patched.modules.model_management.processing_interrupted(): 
                    # self.pbar.close()
                    return x_in

                # batching & compute tiles
                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
                n_rep = len(bboxes)
                ts_tile = self.repeat_tensor(t_in, n_rep)
                cond_tile = self.repeat_tensor(c_crossattn, n_rep)
                c_tile = c_in.copy()
                c_tile['c_crossattn'] = cond_tile
                if 'time_context' in c_in:
                    c_tile['time_context'] = self.repeat_tensor(c_in['time_context'], n_rep)
                for key in c_tile:
                    if key in ['y', 'c_concat']:
                        icond = c_tile[key]
                        if icond.shape[2:] == (self.h, self.w):
                            c_tile[key] = torch.cat([icond[bbox.slicer] for bbox in bboxes])
                        else:
                            c_tile[key] = self.repeat_tensor(icond, n_rep)

                # controlnet tiling
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes))
                if 'control' in c_in:
                    self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control_model'].get_control(x_tile, ts_tile, c_tile, len(cond_or_uncond))

                # stablesr tiling
                # self.switch_stablesr_tensors(batch_id)

                x_tile_out = model_function(x_tile, ts_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                del x_tile_out, x_tile, ts_tile, c_tile

                # update progress bar
                # self.update_pbar()

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)

        return x_out

class MixtureOfDiffusers(AbstractDiffusion):
    """
        Mixture-of-Diffusers Implementation
        https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # weights for custom bboxes
        self.custom_weights: List[Tensor] = []
        self.get_weight = gaussian_weights

    def init_done(self):
        super().init_done()
        # The original gaussian weights can be extremely small, so we rescale them for numerical stability
        self.rescale_factor = 1 / self.weights
        # Meanwhile, we rescale the custom weights in advance to save time of slicing
        for bbox_id, bbox in enumerate(self.custom_bboxes):
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
        # weights for grid bboxes
        # if not hasattr(self, 'tile_weights'):
        # x_in can change sizes cause of ConditioningSetArea, so we have to recalcualte each time
        self.tile_weights = self.get_weight(self.tile_w, self.tile_h)
        return self.tile_weights

    @torch.no_grad()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List= args["cond_or_uncond"]
        c_crossattn: Tensor = c_in['c_crossattn']

        N, C, H, W = x_in.shape

        self.refresh = False
        # self.refresh = True
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # self.pbar = tqdm(total=(self.total_bboxes) * sampling_steps, desc=f"{self.method} Sampling: ")
        # self.pbar = tqdm(total=len(self.batched_bboxes), desc=f"{self.method} Sampling: ")

        # Global sampling
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):     # batch_id is the `Latent tile batch size`
                if ldm_patched.modules.model_management.processing_interrupted(): 
                    # self.pbar.close()
                    return x_in

                # batching
                x_tile_list     = []
                t_tile_list     = []
                icond_map = {}
                # tcond_tile_list = []
                # icond_tile_list = []
                # vcond_tile_list = []
                # control_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                    t_tile_list.append(t_in)
                    if isinstance(c_in, dict):
                        # tcond
                        # tcond_tile = c_crossattn #self.get_tcond(c_in)      # cond, [1, 77, 768]
                        # tcond_tile_list.append(tcond_tile)
                        # present in sdxl
                        for key in ['y', 'c_concat']:
                            if key in c_in:
                                icond=c_in[key] # self.get_icond(c_in)
                                if icond.shape[2:] == (self.h, self.w):
                                    icond = icond[bbox.slicer]
                                if icond_map.get(key, None) is None:
                                    icond_map[key] = []
                                icond_map[key].append(icond)
                        # # vcond:
                        # vcond = self.get_vcond(c_in)
                        # vcond_tile_list.append(vcond)
                    else:
                        print('>> [WARN] not supported, make an issue on github!!')
                n_rep = len(bboxes)
                x_tile      = torch.cat(x_tile_list,     dim=0)          # differs each
                t_tile      = self.repeat_tensor(t_in, n_rep)           # just repeat
                tcond_tile = self.repeat_tensor(c_crossattn, n_rep) # just repeat
                c_tile = c_in.copy()
                c_tile['c_crossattn'] = tcond_tile
                if 'time_context' in c_in:
                    c_tile['time_context'] = self.repeat_tensor(c_in['time_context'], n_rep) # just repeat
                for key in c_tile:
                    if key in ['y', 'c_concat']:
                        icond_tile = torch.cat(icond_map[key], dim=0)  # differs each
                        c_tile[key] = icond_tile
                # vcond_tile = torch.cat(vcond_tile_list, dim=0) if None not in vcond_tile_list else None # just repeat

                # controlnet
                # self.switch_controlnet_tensors(batch_id, N, len(bboxes), is_denoise=True)
                if 'control' in c_in:
                    control=c_in['control']
                    self.process_controlnet(x_tile.shape, x_tile.dtype, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = control.get_control(x_tile, t_tile, c_tile, len(cond_or_uncond))
                
                # stablesr
                # self.switch_stablesr_tensors(batch_id)

                # denoising: here the x is the noise
                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                # de-batching
                for i, bbox in enumerate(bboxes):
                    # These weights can be calcluated in advance, but will cost a lot of vram 
                    # when you have many tiles. So we calculate it here.
                    w = self.tile_weights * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :] * w
                del x_tile_out, x_tile, t_tile, c_tile

                # self.update_pbar()
                # self.pbar.update()
        # self.pbar.close()
        x_out = self.x_buffer

        return x_out


MAX_RESOLUTION=8192
class TiledDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "method": (["MultiDiffusion", "Mixture of Diffusers"], {"default": "Mixture of Diffusers"}),
                                # "tile_width": ("INT", {"default": 96, "min": 16, "max": 256, "step": 16}),
                                "tile_width": ("INT", {"default": 96*opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                                # "tile_height": ("INT", {"default": 96, "min": 16, "max": 256, "step": 16}),
                                "tile_height": ("INT", {"default": 96*opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                                "tile_overlap": ("INT", {"default": 8*opt_f, "min": 0, "max": 256*opt_f, "step": 4*opt_f}),
                                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"

    def apply(self, model: ModelPatcher, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        if method == "Mixture of Diffusers":
            implement = MixtureOfDiffusers()
        else:
            implement = MultiDiffusion()
        
        # if noise_inversion:
        #     get_cache_callback = self.noise_inverse_get_cache
        #     set_cache_callback = None # lambda x0, xt, prompts: self.noise_inverse_set_cache(p, x0, xt, prompts, steps, retouch)
        #     implement.init_noise_inverse(steps, retouch, get_cache_callback, set_cache_callback, renoise_strength, renoise_kernel_size)

        implement.tile_width = tile_width // opt_f
        implement.tile_height = tile_height // opt_f
        implement.tile_overlap = tile_overlap // opt_f
        implement.tile_batch_size = tile_batch_size
        # implement.init_grid_bbox(tile_width, tile_height, tile_overlap, tile_batch_size)
        # # init everything done, perform sanity check & pre-computations
        # implement.init_done()
        # hijack the behaviours
        # implement.hook()
        model = model.clone()
        model.set_model_unet_function_wrapper(implement)
        model.model_options['tiled_diffusion'] = True
        return (model,)
