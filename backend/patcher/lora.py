import torch
import logging
import weakref
from typing import Dict, Any, Tuple, List, Optional, Callable

# Assume these imports work and provide the necessary LoRA loading/parsing functions
import packages_3rdparty.webui_lora_collection.lora as lora_utils_webui
import packages_3rdparty.comfyui_lora_collection.lora as lora_utils_comfyui

from backend import memory_management, utils, operations

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for Patch Types ---
PATCH_TYPE_DIFF = "diff"
PATCH_TYPE_SET = "set"
PATCH_TYPE_LORA = "lora"
PATCH_TYPE_LOKR = "lokr"
PATCH_TYPE_LOHA = "loha"
PATCH_TYPE_GLORA = "glora"
PATCH_TYPE_OFT = "oft"  # Added placeholder for Orthogonal Finetuning


# --- Global Configuration and State ---
extra_weight_calculators: Dict[str, Callable] = {}
lora_collection_priority = [lora_utils_webui, lora_utils_comfyui] # Prioritize which LoRA library functions to use
loaded_lora_cache: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {} # Cache for raw loaded LoRA data {lora_hash: (patch_dict, remaining_dict)}

# --- Utility Functions ---

def get_function(function_name: str) -> Optional[Callable]:
    """
    Retrieves a function implementation from the prioritized LoRA collections.

    Args:
        function_name: The name of the function to retrieve.

    Returns:
        The function object if found, otherwise None.
    """
    for lora_collection in lora_collection_priority:
        if hasattr(lora_collection, function_name):
            func = getattr(lora_collection, function_name)
            # logger.debug(f"Using function '{function_name}' from {lora_collection.__name__}")
            return func
    logger.warning(f"Function '{function_name}' not found in any prioritized LoRA collection.")
    return None

def load_lora(lora_name: str, lora_data: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Loads LoRA data using a function from a prioritized collection, with caching.

    Args:
        lora_name: A unique identifier or hash for the LoRA data (used for caching).
        lora_data: The raw LoRA data or path to load.

    Returns:
        A tuple containing the patch dictionary and any remaining data dictionary,
        or (None, None) if loading fails.
    """
    if lora_name in loaded_lora_cache:
        logger.debug(f"Using cached LoRA data for '{lora_name}'")
        return loaded_lora_cache[lora_name]

    load_func = get_function('load_lora')
    if load_func:
        try:
            patch_dict, remaining_dict = load_func(lora_data, {}) # Assuming second arg is initial state/config
            if patch_dict: # Cache only successful loads
                 # Use weakref to avoid holding strong references if the data is large
                loaded_lora_cache[lora_name] = (weakref.WeakValueDictionary(patch_dict), weakref.WeakValueDictionary(remaining_dict or {}))
            return patch_dict, remaining_dict
        except Exception as e:
            logger.error(f"Error loading LoRA '{lora_name}' using {load_func.__module__}: {e}", exc_info=True)
            return None, None
    else:
        logger.error(f"No 'load_lora' function found to load '{lora_name}'.")
        return None, None

def clear_lora_cache():
    """Clears the loaded LoRA data cache."""
    global loaded_lora_cache
    logger.info("Clearing LoRA data cache.")
    loaded_lora_cache.clear()
    memory_management.soft_empty_cache() # Suggest garbage collection

def inner_str(k: str, prefix: str = "", suffix: str = "") -> str:
    """Helper to extract substring between prefix and suffix."""
    if prefix and k.startswith(prefix):
        k = k[len(prefix):]
    if suffix and k.endswith(suffix):
        k = k[:-len(suffix)]
    return k

# --- Key Mapping Functions ---

def model_lora_keys_clip(model: torch.nn.Module, key_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Generates LoRA key mappings for CLIP models, including T5XXL specific formatting.

    Args:
        model: The CLIP model instance.
        key_map: An optional existing key map to extend.

    Returns:
        The updated key map dictionary.
    """
    key_map = key_map if key_map is not None else {}
    map_func = get_function('model_lora_keys_clip')
    if not map_func:
        logger.error("Cannot generate CLIP LoRA keys: 'model_lora_keys_clip' function not found.")
        return key_map

    model_keys, key_maps = map_func(model, key_map) # Assuming function returns set of keys and the map

    # Add specific T5XXL mapping (example extension)
    for model_key in model_keys:
        if model_key.endswith(".weight"):
            if model_key.startswith("t5xxl.transformer."):
                for prefix in ['te1', 'te2', 'te3']: # Standard prefixes used in some LoRA formats
                    formatted_inner = inner_str(model_key, "t5xxl.transformer.", ".weight")
                    formatted_lora_key = formatted_inner.replace(".", "_")
                    formatted_lora_key = f"lora_{prefix}_{formatted_lora_key}"
                    if formatted_lora_key not in key_map: # Avoid overwriting existing specific maps
                        logger.debug(f"Mapping T5XXL key: {formatted_lora_key} -> {model_key}")
                        key_map[formatted_lora_key] = model_key
            # Potentially add mappings for other CLIP variants here

    return key_maps # Return the map generated/updated by the library function

def model_lora_keys_unet(model: torch.nn.Module, key_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Generates LoRA key mappings for UNet models. Includes placeholder for OFT.

    Args:
        model: The UNet model instance.
        key_map: An optional existing key map to extend.

    Returns:
        The updated key map dictionary.
    """
    key_map = key_map if key_map is not None else {}
    map_func = get_function('model_lora_keys_unet')
    if not map_func:
        logger.error("Cannot generate UNet LoRA keys: 'model_lora_keys_unet' function not found.")
        return key_map

    model_keys, key_maps = map_func(model, key_map)

    # Placeholder: OFT keys might need specific handling or prefixes
    # Example: Search for OFT specific tensor names if the library doesn't handle them
    # oft_keys = {k: v for k, v in model.named_parameters() if 'oft_diag' in k} # Hypothetical
    # for oft_lora_key, model_oft_key in oft_keys.items():
    #     # Construct the LoRA key name expected by OFT files
    #     formatted_oft_key = f"lora_unet_oft_{oft_lora_key.replace('.', '_')}"
    #     if formatted_oft_key not in key_map:
    #         key_map[formatted_oft_key] = model_oft_key
    #         logger.debug(f"Mapping OFT key: {formatted_oft_key} -> {model_oft_key}")

    # TODO: Implement actual OFT key mapping based on common OFT library/format conventions.

    # Placeholder: Block-wise identification could happen here
    # for key in model_keys:
    #     block_name = identify_unet_block(key) # Hypothetical function
    #     # Store block info associated with the key if needed later for filtering
    #     key_map_with_blocks[key] = {'model_key': key_maps.get(key), 'block': block_name}

    return key_maps

# --- Weight Manipulation Functions ---

@torch.inference_mode()
def _apply_dora(
    dora_scale: torch.Tensor,
    weight: torch.nn.Parameter,
    lora_diff: torch.Tensor,
    alpha: float,
    strength: float,
    computation_dtype: torch.dtype
) -> torch.Tensor:
    """
    Applies DORA (Weight-Decomposed Low-Rank Adaptation) to a weight tensor.
    Separated for clarity.

    Args:
        dora_scale: The DoRA scaling factor tensor.
        weight: The original weight tensor (will be modified in-place if strength is 1.0).
        lora_diff: The calculated LoRA difference (e.g., A @ B).
        alpha: The LoRA alpha scaling factor.
        strength: The overall strength factor for the LoRA application.
        computation_dtype: The dtype for intermediate calculations.

    Returns:
        The modified weight tensor.
    """
    # Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_patcher.py#L33
    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, computation_dtype)
    lora_diff = lora_diff * alpha # Apply alpha scaling first

    # Calculate the combined weight (original + scaled LoRA diff)
    weight_calc = weight.to(computation_dtype) + lora_diff.to(computation_dtype)

    # Calculate the norm of the combined weight
    # Ensure float32 for norm calculation stability
    weight_norm = (
        weight_calc.transpose(0, 1)
        .reshape(weight_calc.shape[1], -1)
        .norm(p=2, dim=1, keepdim=True)
        .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
        .transpose(0, 1)
        .to(computation_dtype)
    )
    # Avoid division by zero or near-zero norms
    weight_norm = torch.clamp(weight_norm, min=1e-6)

    # Apply the DORA scaling
    scaled_weight = weight_calc * (dora_scale / weight_norm)

    # Apply strength factor
    if strength == 1.0:
        # Modify weight in-place (or return directly if it's a clone)
        weight[:] = scaled_weight.to(weight.dtype)
        return weight
    else:
        # Calculate the final delta and add it to the original weight
        delta = (scaled_weight - weight.to(computation_dtype)) * strength
        return (weight.to(computation_dtype) + delta).to(weight.dtype)


# --- Core LoRA Merging Logic ---

@torch.inference_mode()
def merge_lora_to_weight(
    patches: List[Tuple],
    weight: torch.nn.Parameter,
    key: str = "unknown_lora_key",
    computation_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Merges a list of LoRA patches onto a single weight tensor.
    Refactored with helper functions for different patch types.

    Args:
        patches: A list of patch tuples. Each tuple typically contains
                 (strength, patch_data, strength_model, offset, function, ...).
        weight: The torch.nn.Parameter tensor to modify.
        key: The key/name associated with this weight (for logging).
        computation_dtype: The dtype to use for intermediate calculations (e.g., float32 for precision).

    Returns:
        The modified weight tensor.
    """
    weight_dtype_backup = weight.dtype
    original_device = weight.device

    # Clone weight only if necessary (computation dtype mismatch or multiple patches)
    # If only one patch and dtypes match, modify in-place might be possible, but cloning is safer.
    if computation_dtype != weight.dtype or len(patches) > 1:
        weight_compute = weight.clone().to(dtype=computation_dtype)
        logger.debug(f"[{key}] Cloning weight for merging. Original dtype: {weight_dtype_backup}, Compute dtype: {computation_dtype}")
    else:
        weight_compute = weight # Operate directly if dtype matches and only one patch (be cautious)

    patch_applied_count = 0
    for i, p in enumerate(patches):
        strength, v, strength_model, offset, function, *maybe_patch_type_info = p
        patch_type = ""

        if function is None:
            function = lambda x: x # Identity function if none provided

        # Determine patch type
        if isinstance(v, list): # Nested patches? (Less common)
            logger.warning(f"[{key}] Nested patch structure detected. Merging recursively.")
            nested_weight = merge_lora_to_weight(v[1:], v[0].clone(), f"{key}_nested", computation_dtype)
            v = (nested_weight,) # Result becomes the data for the outer patch
            patch_type = PATCH_TYPE_DIFF # Assume diff for the result of recursive merge
        elif len(maybe_patch_type_info) > 0 and isinstance(maybe_patch_type_info[0], str):
             # Explicit patch type provided (preferred way)
             patch_type = maybe_patch_type_info[0]
             v = v # v already contains the correct data tuple/list
        elif len(v) == 1:
             patch_type = PATCH_TYPE_DIFF
        elif len(v) == 2 and isinstance(v[0], str): # Old format: ("type", data_tuple)
             patch_type = v[0]
             v = v[1]
        else:
             # Fallback or guess based on content structure (less reliable)
             if isinstance(v, tuple) and len(v) >= 8 and isinstance(v[8], torch.Tensor): patch_type = PATCH_TYPE_LOKR # Check for dora_scale in LoKr tuple
             elif isinstance(v, tuple) and len(v) >= 8 and isinstance(v[7], torch.Tensor): patch_type = PATCH_TYPE_LOHA # Check for dora_scale in LoHa tuple
             elif isinstance(v, tuple) and len(v) >= 5 and isinstance(v[4], torch.Tensor): patch_type = PATCH_TYPE_LORA # Check for dora_scale in LoRA tuple
             # Add more guesses if needed
             else:
                 logger.error(f"[{key}] Could not determine patch type for patch index {i}. Data structure: {type(v)}, len: {len(v) if hasattr(v, '__len__') else 'N/A'}. Skipping patch.")
                 continue

        if strength == 0.0:
            logger.debug(f"[{key}] Skipping patch index {i} (type: {patch_type}) due to strength=0.")
            continue

        logger.debug(f"[{key}] Applying patch index {i}: type={patch_type}, strength={strength}, strength_model={strength_model}, offset={offset}")

        # --- Handle Offset (Sub-tensor patching) ---
        target_weight = weight_compute
        original_full_weight = None # Keep reference if we slice
        if offset is not None:
            try:
                original_full_weight = weight_compute
                dim, start, length = offset
                target_weight = weight_compute.narrow(dim, start, length)
                logger.debug(f"[{key}] Patch applied to slice: dim={dim}, start={start}, length={length}")
            except Exception as e:
                logger.error(f"[{key}] Failed to apply offset {offset} to weight shape {weight_compute.shape}: {e}", exc_info=True)
                continue # Skip this patch if offset is invalid

        # --- Apply Model Strength ---
        if strength_model != 1.0:
            logger.debug(f"[{key}] Applying strength_model {strength_model}")
            target_weight *= strength_model

        # --- Apply Patch Based on Type ---
        try:
            if patch_type == PATCH_TYPE_DIFF:
                w1 = v[0]
                diff = memory_management.cast_to_device(w1, original_device, computation_dtype)
                if diff.shape != target_weight.shape:
                    # Attempt to handle common conv dimension mismatches by padding
                    if diff.ndim == target_weight.ndim == 4 and diff.shape[1] == target_weight.shape[1]: # Check if in_channels match
                        logger.warning(f"[{key}] Patch type '{patch_type}' shape mismatch ({diff.shape} != {target_weight.shape}). Padding output channels.")
                        new_shape = [max(n, m) for n, m in zip(target_weight.shape, diff.shape)]
                        padded_weight = torch.zeros(size=new_shape, dtype=computation_dtype, device=original_device)
                        padded_weight[:target_weight.shape[0], :target_weight.shape[1], :target_weight.shape[2], :target_weight.shape[3]] = target_weight
                        padded_weight[:diff.shape[0], :diff.shape[1], :diff.shape[2], :diff.shape[3]] += strength * function(diff)
                        target_weight = padded_weight # Replace target_weight with the new padded version
                    else:
                         logger.error(f"[{key}] Unhandled shape mismatch for patch type '{patch_type}'. Weight not merged. Target: {target_weight.shape}, Patch: {diff.shape}")
                         continue # Skip patch
                else:
                    target_weight += strength * function(diff.to(target_weight.device)) # Apply function here

            elif patch_type == PATCH_TYPE_SET:
                 # Replace the weight content entirely
                 new_w = memory_management.cast_to_device(v[0], original_device, computation_dtype)
                 if new_w.shape != target_weight.shape:
                      logger.error(f"[{key}] Shape mismatch for patch type '{patch_type}'. Cannot set weight. Target: {target_weight.shape}, Patch: {new_w.shape}")
                      continue
                 target_weight.copy_(function(new_w)) # Apply function before copying

            elif patch_type == PATCH_TYPE_LORA:
                mat1, mat2, alpha_val, conv_weights, dora_scale = v[:5] # LoRA format: (down, up, alpha, Optional[is_conv], Optional[dora_scale])
                mat1 = memory_management.cast_to_device(mat1, original_device, computation_dtype) # LoRA A (Down)
                mat2 = memory_management.cast_to_device(mat2, original_device, computation_dtype) # LoRA B (Up)
                dora_scale = memory_management.cast_to_device(dora_scale, original_device, computation_dtype) if dora_scale is not None else None

                alpha = (alpha_val / mat2.shape[0]) if alpha_val is not None and mat2.shape[0] > 0 else 1.0

                # Handle Conv2D LoRA (rank dropout, etc., handled in library load)
                if len(target_weight.shape) == 4 and conv_weights is not None: # Check dim and if conv data exists
                    # Assume conv_weights is a tuple like (kernel_size, stride, padding) or specific tensors
                    # This part depends heavily on how the LoRA library packs conv LoRA data
                    if len(conv_weights) == 2: # Common case: mat1 is kernel, mat2 is channel matrix
                       kernel_mat = mat1
                       channel_mat = mat2
                       out_c, in_c, H, W = target_weight.shape
                       kernel_c_out, kernel_rank, kH, kW = kernel_mat.shape
                       channel_rank, channel_c_in = channel_mat.shape # Rank x In_channels
                       if kernel_rank != channel_rank:
                            logger.error(f"[{key}] Conv LoRA rank mismatch: Kernel Rank ({kernel_rank}) != Channel Rank ({channel_rank}). Skipping.")
                            continue
                       # Reconstruct the full conv weight diff: Einsum approach
                       # lora_diff = torch.einsum('o r k h, r i -> o i k h', kernel_mat, channel_mat) # Rank maps between kernel and channel
                       # Flattened approach (potentially faster, requires reshape)
                       delta_w = torch.mm(kernel_mat.view(kernel_mat.shape[0], -1), channel_mat).view(out_c, in_c, kH, kW)
                       lora_diff = delta_w

                    elif len(conv_weights) == 3: # LoKr style Conv? (mat2 = up_w, mat3 = up_t)
                       # Similar to lora below, but with mat3
                        mat3 = memory_management.cast_to_device(conv_weights[2], original_device, computation_dtype) # up_t
                        final_shape = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
                        mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1), mat3.transpose(0, 1).flatten(start_dim=1)).reshape(final_shape).transpose(0, 1)
                        lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(target_weight.shape)
                    else:
                        logger.warning(f"[{key}] Unrecognized Conv LoRA data format (conv_weights length: {len(conv_weights)}). Treating as Linear.")
                        lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)).reshape(target_weight.shape) # Fallback to linear mm

                else: # Linear Layer LoRA
                    lora_diff = torch.mm(mat1, mat2) # Standard LoRA: A @ B

                # Ensure diff has the correct shape
                if lora_diff.shape != target_weight.shape:
                    try:
                        lora_diff = lora_diff.reshape(target_weight.shape)
                    except Exception as reshape_err:
                        logger.error(f"[{key}] Cannot reshape LoRA diff from {lora_diff.shape} to target {target_weight.shape}: {reshape_err}. Trying padding.")
                        # Try padding (if output channels differ, like Flux tools)
                        if lora_diff.ndim == target_weight.ndim:
                             if target_weight.shape[0] > lora_diff.shape[0] and all(target_weight.shape[i] == lora_diff.shape[i] for i in range(1, target_weight.ndim)):
                                 pad_amount = target_weight.shape[0] - lora_diff.shape[0]
                                 lora_diff = torch.nn.functional.pad(lora_diff, (0, 0) * (lora_diff.ndim -1) + (0, pad_amount)) # Pad first dim (output channels)
                                 logger.info(f"[{key}] Padded LoRA diff output channels to {target_weight.shape[0]}")
                             elif target_weight.shape[1] > lora_diff.shape[1] and all(target_weight.shape[i] == lora_diff.shape[i] for i in range(target_weight.ndim) if i != 1):
                                 pad_amount = target_weight.shape[1] - lora_diff.shape[1]
                                 lora_diff = torch.nn.functional.pad(lora_diff, (0, 0) * (lora_diff.ndim - 2) + (0, pad_amount, 0, 0)) # Pad second dim (input channels)
                                 logger.info(f"[{key}] Padded LoRA diff input channels to {target_weight.shape[1]}")
                             elif target_weight.shape[0] < lora_diff.shape[0] and all(target_weight.shape[i] == lora_diff.shape[i] for i in range(1, target_weight.ndim)):
                                 logger.warning(f"[{key}] LoRA diff output channels ({lora_diff.shape[0]}) > target ({target_weight.shape[0]}). Truncating diff.")
                                 lora_diff = lora_diff.narrow(0, 0, target_weight.shape[0])
                             elif target_weight.shape[1] < lora_diff.shape[1] and all(target_weight.shape[i] == lora_diff.shape[i] for i in range(target_weight.ndim) if i != 1):
                                 logger.warning(f"[{key}] LoRA diff input channels ({lora_diff.shape[1]}) > target ({target_weight.shape[1]}). Truncating diff.")
                                 lora_diff = lora_diff.narrow(1, 0, target_weight.shape[1])
                             else:
                                 logger.error(f"[{key}] Unhandled shape mismatch after failed reshape and padding attempt. Diff: {lora_diff.shape}, Target: {target_weight.shape}. Skipping.")
                                 continue
                        else:
                            logger.error(f"[{key}] Dimensionality mismatch. Diff: {lora_diff.ndim}D, Target: {target_weight.ndim}D. Skipping.")
                            continue

                # Apply DORA if scale tensor exists
                if dora_scale is not None:
                    logger.debug(f"[{key}] Applying DORA.")
                    # Note: weight_decompose modifies weight in place if strength is 1.0
                    # If target_weight is a slice, this modification happens on the slice view
                    modified_weight = _apply_dora(dora_scale, target_weight, lora_diff, alpha, strength, computation_dtype)
                    if strength != 1.0: # If strength wasn't 1, _apply_dora returned a new tensor
                         target_weight = modified_weight # Update target_weight reference
                else:
                    # Standard LoRA addition
                    delta = (strength * alpha) * lora_diff
                    target_weight += function(delta).to(target_weight.dtype)


            elif patch_type == PATCH_TYPE_LOKR:
                # Lokr format: (w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2, dora_scale)
                w1, w2, alpha_val, w1_a, w1_b, w2_a, w2_b, t2, dora_scale = v[:9]
                dora_scale = memory_management.cast_to_device(dora_scale, original_device, computation_dtype) if dora_scale is not None else None
                dim = None # Rank dimension

                # Reconstruct w1 if decomposed
                if w1 is None:
                    w1a = memory_management.cast_to_device(w1_a, original_device, computation_dtype)
                    w1b = memory_management.cast_to_device(w1_b, original_device, computation_dtype)
                    w1 = torch.mm(w1a, w1b)
                    dim = w1b.shape[0] # Rank is inner dim of decomposition
                else:
                    w1 = memory_management.cast_to_device(w1, original_device, computation_dtype)

                # Reconstruct w2 if decomposed
                if w2 is None:
                    w2a = memory_management.cast_to_device(w2_a, original_device, computation_dtype)
                    w2b = memory_management.cast_to_device(w2_b, original_device, computation_dtype)
                    if dim is None: dim = w2b.shape[0] # Get rank if not already set
                    elif dim != w2b.shape[0]: logger.warning(f"[{key}] LoKr rank mismatch between w1 ({dim}) and w2 ({w2b.shape[0]}) decomposition.")

                    if t2 is None: # Standard decomposition
                        w2 = torch.mm(w2a, w2b)
                    else: # CP decomposition (tensor contraction)
                         t2_tensor = memory_management.cast_to_device(t2, original_device, computation_dtype)
                         # Contract t2 with w2_a and w2_b
                         # This einsum depends on the exact shape/convention of t2, w2a, w2b
                         # Example: 'i j k l, j r, i p -> p r k l' assumes t2 is rank x rank x K x L, w2a/b project features
                         try:
                             w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2_tensor, w2b, w2a)
                             logger.debug(f"[{key}] LoKr using CP decomposition (t2).")
                         except Exception as einsum_err:
                             logger.error(f"[{key}] LoKr CP decomposition einsum failed: {einsum_err}. Shapes: t2={t2_tensor.shape}, w2a={w2a.shape}, w2b={w2b.shape}. Skipping.", exc_info=True)
                             continue
                else:
                    w2 = memory_management.cast_to_device(w2, original_device, computation_dtype)

                # Prepare for Kronecker product
                if len(target_weight.shape) == 4 and len(w1.shape) == 2 and len(w2.shape) == 2:
                     # Common case for Conv2D: w1 affects output channels, w2 affects input channels/kernel
                     # kron(w1, w2) computation needs careful shape handling
                     # w1 needs shape (OutA, OutB), w2 needs shape (InA, InB)
                     # Resulting kron has shape (OutA*InA, OutB*InB)
                     # Need to reshape this to match the target conv weight (OutC, InC, kH, kW)
                     # This requires knowing how w1/w2 map to OutC/InC and kH/kW
                     logger.debug(f"[{key}] Applying LoKr to Conv layer.")
                     w1 = w1.unsqueeze(2).unsqueeze(3) # Add dummy spatial dims for broadcasting in kron? Or handle reshape later.
                     w2 = w2.unsqueeze(0).unsqueeze(1) # Add dummy output dims? Check convention.
                     # This part is tricky and needs verification based on LoKr library's conv implementation
                     # Let's assume a simpler reshape target for now
                     lora_diff = torch.kron(w1, w2) # Calculate Kronecker product
                     if lora_diff.numel() != target_weight.numel():
                         logger.error(f"[{key}] LoKr Kronecker product result size ({lora_diff.numel()}) != target weight size ({target_weight.numel()}). Skipping.")
                         continue
                     lora_diff = lora_diff.reshape(target_weight.shape) # Reshape to target
                elif len(target_weight.shape) == 2 and len(w1.shape) == 2 and len(w2.shape) == 2: # Linear
                    lora_diff = torch.kron(w1, w2)
                    if lora_diff.shape != target_weight.shape:
                         if lora_diff.numel() == target_weight.numel():
                             logger.warning(f"[{key}] Reshaping LoKr diff from {lora_diff.shape} to target {target_weight.shape}.")
                             lora_diff = lora_diff.reshape(target_weight.shape)
                         else:
                             logger.error(f"[{key}] LoKr Kronecker product result shape ({lora_diff.shape}) != target weight shape ({target_weight.shape}). Size mismatch. Skipping.")
                             continue
                else:
                     logger.error(f"[{key}] Unsupported tensor dimensions for LoKr. Target: {target_weight.ndim}D, W1: {w1.ndim}D, W2: {w2.ndim}D. Skipping.")
                     continue

                alpha = (alpha_val / dim) if alpha_val is not None and dim is not None and dim > 0 else 1.0

                if dora_scale is not None:
                     logger.debug(f"[{key}] Applying DORA to LoKr.")
                     modified_weight = _apply_dora(dora_scale, target_weight, lora_diff, alpha, strength, computation_dtype)
                     if strength != 1.0: target_weight = modified_weight
                else:
                     delta = (strength * alpha) * lora_diff
                     target_weight += function(delta).to(target_weight.dtype)


            elif patch_type == PATCH_TYPE_LOHA:
                # Loha format: (w1a, w1b, alpha, w2a, w2b, t1, t2, dora_scale)
                w1a, w1b, alpha_val, w2a, w2b, t1, t2, dora_scale = v[:8]
                dora_scale = memory_management.cast_to_device(dora_scale, original_device, computation_dtype) if dora_scale is not None else None
                dim = w1b.shape[0] # Rank

                alpha = (alpha_val / dim) if alpha_val is not None and dim > 0 else 1.0

                w1a = memory_management.cast_to_device(w1a, original_device, computation_dtype)
                w1b = memory_management.cast_to_device(w1b, original_device, computation_dtype)
                w2a = memory_management.cast_to_device(w2a, original_device, computation_dtype)
                w2b = memory_management.cast_to_device(w2b, original_device, computation_dtype)

                if t1 is not None and t2 is not None: # CP decomposition
                    logger.debug(f"[{key}] LoHa using CP decomposition (t1, t2).")
                    t1_tensor = memory_management.cast_to_device(t1, original_device, computation_dtype)
                    t2_tensor = memory_management.cast_to_device(t2, original_device, computation_dtype)
                    try:
                         # Einsum depends on tensor shapes/convention (similar to LoKr)
                        m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1_tensor, w1b, w1a)
                        m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2_tensor, w2b, w2a)
                    except Exception as einsum_err:
                        logger.error(f"[{key}] LoHa CP decomposition einsum failed: {einsum_err}. Skipping.", exc_info=True)
                        continue
                else: # Standard decomposition
                    m1 = torch.mm(w1a, w1b)
                    m2 = torch.mm(w2a, w2b)

                # Hadamard product
                lora_diff = m1 * m2
                if lora_diff.shape != target_weight.shape:
                    try:
                        lora_diff = lora_diff.reshape(target_weight.shape)
                    except Exception as reshape_err:
                        logger.error(f"[{key}] Cannot reshape LoHa diff from {lora_diff.shape} to target {target_weight.shape}: {reshape_err}. Skipping.")
                        continue

                if dora_scale is not None:
                    logger.debug(f"[{key}] Applying DORA to LoHa.")
                    modified_weight = _apply_dora(dora_scale, target_weight, lora_diff, alpha, strength, computation_dtype)
                    if strength != 1.0: target_weight = modified_weight
                else:
                    delta = (strength * alpha) * lora_diff
                    target_weight += function(delta).to(target_weight.dtype)


            elif patch_type == PATCH_TYPE_GLORA:
                # GLoRA format: (a1, a2, b1, b2, alpha, dora_scale)
                a1, a2, b1, b2, alpha_val, dora_scale = v[:6]
                dora_scale = memory_management.cast_to_device(dora_scale, original_device, computation_dtype) if dora_scale is not None else None
                # Determine rank (can be ambiguous, often based on a specific dim)
                rank = a1.shape[1] # A common convention
                alpha = (alpha_val / rank) if alpha_val is not None and rank > 0 else 1.0

                a1 = memory_management.cast_to_device(a1, original_device, computation_dtype)
                a2 = memory_management.cast_to_device(a2, original_device, computation_dtype)
                b1 = memory_management.cast_to_device(b1, original_device, computation_dtype)
                b2 = memory_management.cast_to_device(b2, original_device, computation_dtype)

                # GLoRA calculation: W' = W @ A1 @ A2 + B1 @ B2
                if target_weight.dim() > 2: # Conv or other ND layers
                    # Need to handle potential dimension differences
                    # Assume einsum is needed for correct application
                    # Example: 'o i ..., i j -> o j ...' applies first matrix, then second
                    try:
                         intermediate = torch.einsum("o i ..., i j -> o j ...", target_weight, a1)
                         proj_diff = torch.einsum("o j ..., j k -> o k ...", intermediate, a2)
                    except Exception as e:
                         logger.error(f"[{key}] GLoRA projection einsum failed: {e}. Shapes: W={target_weight.shape}, a1={a1.shape}, a2={a2.shape}", exc_info=True)
                         continue
                else: # Linear layer
                    proj_diff = torch.mm(torch.mm(target_weight, a1), a2)

                bias_diff = torch.mm(b1, b2)

                # Combine and reshape
                lora_diff = (proj_diff - target_weight) + bias_diff # Calculate the effective delta
                if lora_diff.shape != target_weight.shape:
                     try:
                         lora_diff = lora_diff.reshape(target_weight.shape)
                     except Exception as reshape_err:
                         logger.error(f"[{key}] Cannot reshape GLoRA diff from {lora_diff.shape} to target {target_weight.shape}: {reshape_err}. Skipping.")
                         continue

                if dora_scale is not None:
                    logger.debug(f"[{key}] Applying DORA to GLoRA.")
                    # Apply DORA to the *delta*, not the full weight
                    modified_weight = _apply_dora(dora_scale, target_weight, lora_diff, alpha, strength, computation_dtype)
                    if strength != 1.0: target_weight = modified_weight
                else:
                    delta = (strength * alpha) * lora_diff
                    target_weight += function(delta).to(target_weight.dtype)

            elif patch_type == PATCH_TYPE_OFT:
                 # Placeholder for OFT implementation
                 # OFT typically involves rotating weights using orthogonal matrices (often block-diagonal)
                 # Example: R = block_diag(R1, R2, ...), W' = W @ R
                 # Need OFT-specific parameters (R matrix or its decomposition) from 'v'
                 # R = v[0] # Hypothetical: R is the orthogonal matrix
                 # R = memory_management.cast_to_device(R, original_device, computation_dtype)
                 # if R.shape[0] != target_weight.shape[1] or R.shape[1] != target_weight.shape[1]:
                 #      logger.error(f"[{key}] OFT matrix R shape {R.shape} incompatible with weight's input dim {target_weight.shape[1]}. Skipping.")
                 #      continue
                 # # Apply rotation (simple matrix multiply for linear, may need einsum for conv)
                 # if target_weight.dim() == 2:
                 #     rotated_weight = torch.mm(target_weight, R)
                 # else: # Conv/other ND
                 #     # rotated_weight = torch.einsum("o i ..., i j -> o j ...", target_weight, R) # Example
                 #     logger.warning(f"[{key}] OFT for {target_weight.dim()}D tensors not fully implemented. Using fallback linear multiply.")
                 #     out_dim, in_dim = target_weight.shape[:2]
                 #     orig_shape = target_weight.shape
                 #     rotated_weight = torch.mm(target_weight.view(out_dim, -1), R).view(orig_shape) # May not be correct

                 # lora_diff = rotated_weight - target_weight # Calculate delta
                 # delta = strength * function(lora_diff) # OFT might not use alpha/dora usually
                 # target_weight += delta.to(target_weight.dtype)
                 logger.warning(f"[{key}] Patch type '{patch_type}' is not yet implemented. Skipping patch.")
                 continue # Skip OFT until implemented


            elif patch_type in extra_weight_calculators:
                 # Custom patch types registered externally
                 logger.debug(f"[{key}] Using extra weight calculator for patch type '{patch_type}'.")
                 target_weight = extra_weight_calculators[patch_type](target_weight, strength, v, function, computation_dtype, original_device)


            else:
                 logger.warning(f"[{key}] Patch type '{patch_type}' not recognized. Skipping patch.")
                 continue

            patch_applied_count += 1

        except Exception as e:
            logger.error(f"[{key}] ERROR applying patch type '{patch_type}' (index {i}): {e}", exc_info=True)
            # Decide whether to continue with next patch or raise error
            # For robustness, let's try to continue
            continue # Skip to next patch for this weight

        finally:
            # --- Restore Full Weight if Offset Was Used ---
            if original_full_weight is not None:
                 # Ensure the modified slice is written back to the original tensor context
                 # This should happen automatically if target_weight was a view,
                 # but explicit assignment can be safer if target_weight was replaced (e.g., padding)
                 if target_weight is not original_full_weight.narrow(offset[0], offset[1], offset[2]):
                     try:
                         original_full_weight.narrow(offset[0], offset[1], offset[2]).copy_(target_weight)
                         logger.debug(f"[{key}] Copied modified slice back to original weight tensor.")
                     except Exception as copy_err:
                         logger.error(f"[{key}] Error copying modified slice back: {copy_err}", exc_info=True)
                 # Restore weight_compute to the full tensor for the next patch iteration
                 weight_compute = original_full_weight


    if patch_applied_count > 0:
        logger.info(f"[{key}] Applied {patch_applied_count} / {len(patches)} patches successfully.")
    else:
        logger.warning(f"[{key}] No patches were successfully applied out of {len(patches)}.")


    # --- Final Conversion and Return ---
    if weight_compute.dtype != weight_dtype_backup:
        logger.debug(f"[{key}] Converting final weight back to original dtype: {weight_dtype_backup}")
        weight_compute = weight_compute.to(dtype=weight_dtype_backup)

    # If the original weight was modified in-place, return it.
    # If a clone was made, return the modified clone.
    return weight_compute


# --- Device Management Utilities ---

def get_parameter_devices(model: torch.nn.Module) -> Dict[str, torch.device]:
    """Stores the device placement for each model parameter."""
    parameter_devices = {}
    for key, p in model.named_parameters():
        parameter_devices[key] = p.device
    logger.debug(f"Recorded devices for {len(parameter_devices)} parameters.")
    return parameter_devices

def set_parameter_devices(model: torch.nn.Module, parameter_devices: Dict[str, torch.device], force_move: bool = False):
    """
    Restores the device placement for model parameters.

    Args:
        model: The model to modify.
        parameter_devices: Dictionary mapping parameter keys to target devices.
        force_move: If True, move parameters even if they are already on the target device (useful after potential in-place ops).
    """
    logger.info(f"Setting devices for {len(parameter_devices)} parameters...")
    params_moved = 0
    params_checked = 0
    with torch.no_grad(): # Ensure no gradients are computed during device moves
        for key, target_device in parameter_devices.items():
            try:
                # Use utils.get_attr for robustness with nested modules
                current_param = utils.get_attr(model, key)
                if current_param is None:
                    logger.warning(f"Parameter '{key}' not found in model during device setting. Skipping.")
                    continue

                params_checked += 1
                if force_move or current_param.device != target_device:
                    # Move the parameter and wrap it back into nn.Parameter
                    new_param = torch.nn.Parameter(current_param.to(device=target_device, non_blocking=True), requires_grad=False) # Assume inference
                    # Use utils.set_attr_raw for direct replacement
                    utils.set_attr_raw(model, key, new_param)
                    params_moved += 1
                # else: Parameter already on correct device
            except AttributeError:
                logger.warning(f"Attribute error getting/setting parameter '{key}'. It might have been dynamically removed or is not a Parameter.")
            except Exception as e:
                 logger.error(f"Error setting device for parameter '{key}' to {target_device}: {e}", exc_info=True)

    logger.info(f"Device setting complete. Checked: {params_checked}, Moved: {params_moved}")
    memory_management.soft_empty_cache() # Clean up potential memory fragments
    return model


# --- LoRA Loader Class ---

class LoraLoader:
    """
    Manages the application and removal of LoRA patches to a model,
    handling backups, online/offline modes, quantization, and caching.
    """
    def __init__(self, model: torch.nn.Module, model_name: str = "model"):
        self.model_ref = weakref.ref(model) # Use weak reference to avoid circular dependencies
        self.model_name = model_name # For logging
        self.backup: Dict[str, torch.Tensor] = {} # Stores original weights {key: tensor_on_offload_device}
        self.online_lora_layers: weakref.WeakSet[torch.nn.Module] = weakref.WeakSet() # Layers with active online LoRAs
        self.loaded_lora_config_hash: Optional[str] = None # Hash of the applied LoRA configuration string
        logger.info(f"LoraLoader initialized for '{self.model_name}'.")

    def _get_model(self) -> Optional[torch.nn.Module]:
        """Safely retrieves the model from the weak reference."""
        model = self.model_ref()
        if model is None:
            logger.error(f"LoraLoader: Model for '{self.model_name}' no longer exists!")
            return None
        return model

    def _clear_online_loras(self, model: torch.nn.Module):
        """Removes online LoRA hooks/attributes from layers."""
        if not self.online_lora_layers:
            return

        logger.info(f"Clearing online LoRA attributes from {len(self.online_lora_layers)} layers...")
        count = 0
        # Iterate over a copy since the set might change if layers are deleted
        layers_to_clear = list(self.online_lora_layers)
        for layer in layers_to_clear:
            if hasattr(layer, 'forge_online_loras'):
                del layer.forge_online_loras
                count += 1
            # Remove the layer from the tracking set
            # This happens automatically due to WeakSet semantics if layer is deleted elsewhere,
            # but we can explicitly discard it here too.
            self.online_lora_layers.discard(layer)

        # Clear the set itself after iteration
        self.online_lora_layers.clear()
        logger.info(f"Cleared online LoRA attributes from {count} layers.")


    def _restore_weights(self, model: torch.nn.Module, parameter_devices: Dict[str, torch.device]):
        """Restores original weights from the backup."""
        if not self.backup:
            logger.info("No backed-up weights to restore.")
            return

        logger.info(f"Restoring {len(self.backup)} weights from backup...")
        restored_count = 0
        with torch.no_grad():
            for key, backed_up_weight in self.backup.items():
                try:
                    target_device = parameter_devices.get(key)
                    if target_device is None:
                        logger.warning(f"Cannot determine target device for backed-up weight '{key}'. Skipping restore.")
                        continue

                    # Ensure backup is on the correct device before setting
                    restored_weight = backed_up_weight.to(target_device)

                    # Check if the weight needs quantization restoration (BNB/GGUF)
                    # This requires storing metadata about quantization during backup, which isn't done yet.
                    # Simple restoration for now:
                    current_param = utils.get_attr(model, key)
                    if current_param is None:
                        logger.warning(f"Cannot find parameter '{key}' in model to restore. Skipping.")
                        continue

                    # Check if the current param is quantized (and backup wasn't)
                    is_bnb = hasattr(current_param, 'bnb_quantized') and operations.bnb_avaliable
                    is_gguf = hasattr(current_param, 'gguf_cls')

                    if is_bnb:
                        parent_layer, child_key, _ = utils.get_attr_with_parent(model, key)
                        if hasattr(parent_layer, 'reload_weight'):
                            logger.debug(f"Restoring '{key}' by reloading BNB layer.")
                            parent_layer.reload_weight(restored_weight)
                            restored_count += 1
                        else:
                             logger.warning(f"Cannot restore quantized BNB weight '{key}': Parent layer lacks 'reload_weight' method.")
                    elif is_gguf:
                        gguf_cls = getattr(current_param, 'gguf_cls', None)
                        gguf_param = current_param # The quantized param itself
                        if gguf_cls and hasattr(gguf_cls, 'quantize_pytorch'):
                            logger.debug(f"Restoring '{key}' by re-quantizing GGUF layer.")
                            gguf_cls.quantize_pytorch(restored_weight, gguf_param)
                            restored_count += 1
                        else:
                            logger.warning(f"Cannot restore quantized GGUF weight '{key}': Missing 'gguf_cls' or 'quantize_pytorch' method.")
                    else:
                        # Standard parameter restore
                        new_param = torch.nn.Parameter(restored_weight, requires_grad=False)
                        utils.set_attr_raw(model, key, new_param)
                        restored_count += 1

                except Exception as e:
                    logger.error(f"Error restoring weight '{key}': {e}", exc_info=True)

        logger.info(f"Weight restoration complete. Restored {restored_count} weights.")
        # Clear backup after successful restoration
        self.backup.clear()


    @torch.inference_mode()
    def refresh(
        self,
        lora_patch_configs: Dict[str, Tuple[float, float, float, bool, str]], # {lora_name: (weight, clip_w, unet_w, online_mode, lora_hash_or_path)}
        offload_device: torch.device = torch.device('cpu'),
        force_refresh: bool = False,
        computation_dtype: torch.dtype = torch.float32
    ):
        """
        Applies or updates LoRA patches based on the provided configurations.

        Args:
            lora_patch_configs: Dictionary where keys are LoRA identifiers and values are tuples:
                (strength: float, strength_clip: float, strength_unet: float, online_mode: bool, lora_id: str).
                'lora_id' is used for loading/caching the actual LoRA data. Strength values multiply the base LoRA strength.
            offload_device: Device to store weight backups (usually CPU).
            force_refresh: If True, forces reloading and patching even if the config hash matches.
            computation_dtype: Dtype for intermediate patch calculations.
        """
        model = self._get_model()
        if model is None: return

        # Generate a hash representing the current LoRA configuration
        # Sort by name to ensure consistent hash regardless of dict order
        config_items = sorted(lora_patch_configs.items())
        current_config_hash = str(config_items)

        if current_config_hash == self.loaded_lora_config_hash and not force_refresh:
            logger.info(f"LoRA configuration hasn't changed ({current_config_hash[:20]}...). Skipping refresh.")
            return

        logger.info(f"Refreshing LoRA patches for '{self.model_name}'. Force refresh: {force_refresh}")
        logger.debug(f"New LoRA config hash: {current_config_hash[:50]}...")
        if self.loaded_lora_config_hash:
             logger.debug(f"Old LoRA config hash: {self.loaded_lora_config_hash[:50]}...")


        # --- Preparation ---
        memory_management.signal_empty_cache() # Hint that we might need memory
        original_parameter_devices = get_parameter_devices(model) # Store original devices


        # --- Restore Model State ---
        # 1. Clear any existing online LoRA hooks/data
        self._clear_online_loras(model)

        # 2. Restore weights from backup
        # Need original devices to restore weights correctly before potentially moving them again
        self._restore_weights(model, original_parameter_devices)

        # 3. Ensure parameters are back on their original devices after potential restoration moves
        set_parameter_devices(model, original_parameter_devices, force_move=True) # Force ensures correct placement

        # --- Load and Merge LoRA Patches ---
        # { (target_model_key, is_online_mode): [patch1_tuple, patch2_tuple, ...] }
        all_patches_merged: Dict[Tuple[str, bool], List[Tuple]] = {}
        key_map_clip = None # Lazy load key maps
        key_map_unet = None

        loras_to_load = list(lora_patch_configs.items())
        logger.info(f"Processing {len(loras_to_load)} LoRA configurations.")

        for lora_name, (strength, strength_clip, strength_unet, online_mode, lora_id) in loras_to_load:
            if strength_clip == 0.0 and strength_unet == 0.0:
                logger.debug(f"Skipping LoRA '{lora_name}' (ID: {lora_id}) as both CLIP and UNet strengths are 0.")
                continue

            # Load LoRA data (using cache)
            patch_dict, _ = load_lora(lora_id, lora_id) # Assume lora_id is path or data loadable by load_lora

            if patch_dict is None:
                logger.error(f"Failed to load LoRA data for '{lora_name}' (ID: {lora_id}). Skipping this LoRA.")
                continue

            logger.info(f"Applying LoRA '{lora_name}' (ID: {lora_id}) with strength={strength}, clip={strength_clip}, unet={strength_unet}, online={online_mode}")

            # Generate key maps if not already done
            # This assumes the model has distinct 'clip' and 'unet' parts detectable by the key functions
            # TODO: Make model type detection more robust if needed
            if key_map_clip is None: key_map_clip = model_lora_keys_clip(model)
            if key_map_unet is None: key_map_unet = model_lora_keys_unet(model)

            # Distribute patches based on keys and strengths
            for lora_key, patch_data_list in patch_dict.items():
                target_model_key = None
                lora_strength_multiplier = 0.0

                # Determine if the key belongs to CLIP or UNet
                # This logic might need refinement based on key naming conventions
                if lora_key in key_map_clip:
                    target_model_key = key_map_clip[lora_key]
                    lora_strength_multiplier = strength_clip
                elif lora_key in key_map_unet:
                    target_model_key = key_map_unet[lora_key]
                    lora_strength_multiplier = strength_unet
                # Add checks for other model parts (e.g., VAE) if necessary
                else:
                    # Attempt fallback mapping (e.g., direct key match if no prefix matched)
                    if lora_key in original_parameter_devices:
                        target_model_key = lora_key
                        # Decide a default strength or use the main strength?
                        lora_strength_multiplier = strength
                        logger.warning(f"LoRA key '{lora_key}' not found in CLIP/UNet maps, attempting direct match. Using base strength {strength}.")
                    else:
                        logger.warning(f"LoRA key '{lora_key}' from '{lora_name}' does not map to any known model key. Skipping this key.")
                        continue

                if lora_strength_multiplier == 0.0:
                    # logger.debug(f"Skipping key '{lora_key}' -> '{target_model_key}' due to zero strength multiplier.")
                    continue

                # Prepare the final patch tuple(s) with adjusted strength
                # patch_data_list is usually [(lora_strength, patch_content_tuple)]
                for single_patch_data in patch_data_list:
                    base_lora_strength = single_patch_data[0]
                    final_strength = base_lora_strength * lora_strength_multiplier
                    # Reconstruct the patch tuple with the final strength
                    # Assumes original tuple structure: (strength, data, strength_model, offset, function, ...)
                    final_patch_tuple = (final_strength,) + single_patch_data[1:]

                    merge_key = (target_model_key, online_mode)
                    if merge_key not in all_patches_merged:
                        all_patches_merged[merge_key] = []
                    all_patches_merged[merge_key].append(final_patch_tuple)
                    # logger.debug(f"Scheduled patch for '{target_model_key}' (Online: {online_mode}) from LoRA '{lora_name}', key '{lora_key}', final strength {final_strength:.3f}")


        # --- Apply Patches ---
        logger.info(f"Applying merged patches to {len(all_patches_merged)} target model keys...")
        applied_offline_keys = set()

        for (target_key, online_mode), current_patches in all_patches_merged.items():
            try:
                parent_layer, child_key, weight_param = utils.get_attr_with_parent(model, target_key)
                if not isinstance(weight_param, torch.nn.Parameter):
                     logger.warning(f"Target '{target_key}' is not a torch.nn.Parameter. Skipping patches.")
                     continue
            except Exception as e:
                 logger.error(f"Error accessing model parameter '{target_key}': {e}. Skipping patches for this key.", exc_info=True)
                 continue

            if online_mode:
                 # --- Handle Online LoRA ---
                 logger.debug(f"Applying {len(current_patches)} patches to '{target_key}' in online mode.")
                 if not hasattr(parent_layer, 'forge_online_loras'):
                     parent_layer.forge_online_loras = {} # Initialize storage on the layer

                 # Store the list of patch tuples directly
                 parent_layer.forge_online_loras[child_key] = current_patches
                 self.online_lora_layers.add(parent_layer) # Track layers with online LoRAs

            else:
                 # --- Handle Offline LoRA (Direct Weight Modification) ---
                 logger.debug(f"Applying {len(current_patches)} patches to '{target_key}' in offline mode.")
                 applied_offline_keys.add(target_key)

                 # 1. Backup original weight if not already done in this refresh cycle
                 if target_key not in self.backup:
                     # Move backup to offload device immediately to save VRAM
                     self.backup[target_key] = weight_param.data.clone().to(device=offload_device)
                     logger.debug(f"Backed up '{target_key}' to {offload_device}.")

                 # 2. Handle Quantization (Get Dequantized Weight)
                 original_weight = weight_param.data # Work with the tensor data
                 is_bnb = False
                 is_gguf = False
                 bnb_layer_info = None
                 gguf_info = None

                 if hasattr(weight_param, 'bnb_quantized') and operations.bnb_avaliable:
                     is_bnb = True
                     bnb_layer_info = (parent_layer, child_key) # Need parent layer to reload later
                     logger.debug(f"Dequantizing BNB weight for '{target_key}'.")
                     try:
                         from backend.operations_bnb import functional_dequantize_4bit
                         original_weight = functional_dequantize_4bit(weight_param)
                     except Exception as dequant_err:
                          logger.error(f"Failed to dequantize BNB weight '{target_key}': {dequant_err}. Skipping patch application.", exc_info=True)
                          # Remove backup if we can't process it
                          if target_key in self.backup: del self.backup[target_key]
                          continue

                 elif hasattr(weight_param, 'gguf_cls'):
                     is_gguf = True
                     gguf_info = (getattr(weight_param, 'gguf_cls'), weight_param) # Need class and original param
                     logger.debug(f"Dequantizing GGUF weight for '{target_key}'.")
                     try:
                         from backend.operations_gguf import dequantize_tensor
                         original_weight = dequantize_tensor(weight_param)
                     except Exception as dequant_err:
                         logger.error(f"Failed to dequantize GGUF weight '{target_key}': {dequant_err}. Skipping patch application.", exc_info=True)
                         if target_key in self.backup: del self.backup[target_key]
                         continue

                 # 3. Merge LoRA patches onto the (potentially dequantized) weight
                 modified_weight = None
                 try:
                     # Ensure original_weight is a parameter-like tensor for merge function
                     # We use the data, but merge_lora expects Parameter-like behavior
                     temp_param_for_merge = torch.nn.Parameter(original_weight, requires_grad=False)
                     modified_weight = merge_lora_to_weight(
                         current_patches,
                         temp_param_for_merge, # Pass the temporary Parameter
                         target_key,
                         computation_dtype=computation_dtype
                     )
                 except torch.cuda.OutOfMemoryError as oom_err:
                     logger.warning(f"Out of memory applying LoRA to '{target_key}'. Attempting offload retry: {oom_err}")
                     memory_management.soft_empty_cache()
                     # Offload the entire model temporarily
                     temp_offload_devices = {k: offload_device for k in original_parameter_devices}
                     set_parameter_devices(model, temp_offload_devices)
                     memory_management.soft_empty_cache()
                     try:
                         # Ensure weight is on CPU for merge if model offloaded
                         temp_param_for_merge = torch.nn.Parameter(original_weight.to(offload_device), requires_grad=False)
                         modified_weight = merge_lora_to_weight(
                             current_patches,
                             temp_param_for_merge,
                             target_key,
                             computation_dtype=computation_dtype # Keep compute dtype
                         )
                         modified_weight = modified_weight.to(original_parameter_devices[target_key]) # Move result back
                     except Exception as retry_err:
                         logger.error(f"Offload retry failed for '{target_key}': {retry_err}. Skipping patch application.", exc_info=True)
                         # Restore original weight if backup exists
                         if target_key in self.backup:
                             original_weight_restored = self.backup[target_key].to(original_parameter_devices[target_key])
                             utils.set_attr_raw(model, target_key, torch.nn.Parameter(original_weight_restored, requires_grad=False))
                             del self.backup[target_key] # Remove useless backup
                         # Ensure model parameters are moved back after failed attempt
                         set_parameter_devices(model, original_parameter_devices)
                         continue # Skip this key
                     finally:
                         # Ensure model parameters are moved back even after successful retry
                         set_parameter_devices(model, original_parameter_devices)
                 except Exception as merge_err:
                     logger.error(f"Failed to merge LoRA patches onto weight '{target_key}': {merge_err}. Skipping patch application.", exc_info=True)
                     if target_key in self.backup: del self.backup[target_key]
                     continue

                 # Ensure modified_weight exists
                 if modified_weight is None:
                     logger.error(f"LoRA merge result for '{target_key}' is None. Skipping application.")
                     if target_key in self.backup: del self.backup[target_key]
                     continue

                 # 4. Apply Modified Weight (and handle Re-quantization)
                 try:
                     if is_bnb and bnb_layer_info:
                         logger.debug(f"Re-quantizing BNB weight for '{target_key}'.")
                         bnb_parent_layer, _ = bnb_layer_info
                         if hasattr(bnb_parent_layer, 'reload_weight'):
                             # Ensure weight is on the correct device before reloading
                             target_device = original_parameter_devices.get(target_key)
                             bnb_parent_layer.reload_weight(modified_weight.to(target_device))
                         else:
                             logger.error(f"Cannot re-quantize BNB weight '{target_key}': Parent layer lacks 'reload_weight' method.")
                     elif is_gguf and gguf_info:
                         logger.debug(f"Re-quantizing GGUF weight for '{target_key}'.")
                         gguf_cls, gguf_param = gguf_info
                         if gguf_cls and hasattr(gguf_cls, 'quantize_pytorch'):
                             target_device = original_parameter_devices.get(target_key)
                             gguf_cls.quantize_pytorch(modified_weight.to(target_device), gguf_param) # Quantize in-place onto original param obj
                         else:
                             logger.error(f"Cannot re-quantize GGUF weight '{target_key}': Missing 'gguf_cls' or 'quantize_pytorch' method.")
                     else:
                         # Standard parameter update
                         target_device = original_parameter_devices.get(target_key)
                         new_param = torch.nn.Parameter(modified_weight.to(target_device), requires_grad=False)
                         utils.set_attr_raw(model, target_key, new_param)
                 except Exception as apply_err:
                     logger.error(f"Failed to apply modified weight or re-quantize '{target_key}': {apply_err}. State may be inconsistent.", exc_info=True)
                     # Consider restoring original weight here as well
                     if target_key in self.backup:
                         try:
                             original_weight_restored = self.backup[target_key].to(original_parameter_devices[target_key])
                             utils.set_attr_raw(model, target_key, torch.nn.Parameter(original_weight_restored, requires_grad=False))
                         except Exception as restore_err:
                              logger.error(f"Failed to restore original weight for '{target_key}' after apply error: {restore_err}")
                         del self.backup[target_key] # Remove backup


        # --- Finalization ---
        # Ensure all parameters are on their original devices after all patching
        set_parameter_devices(model, original_parameter_devices, force_move=True) # Force move to be sure

        # Clean up backups for keys that were not touched by offline LoRAs in this run
        keys_to_unbackup = set(self.backup.keys()) - applied_offline_keys
        if keys_to_unbackup:
             logger.debug(f"Removing stale backups for {len(keys_to_unbackup)} keys not patched offline in this run.")
             for key in keys_to_unbackup:
                 del self.backup[key]

        self.loaded_lora_config_hash = current_config_hash # Update hash to reflect applied config
        memory_management.soft_empty_cache() # Final cleanup
        logger.info(f"LoRA refresh completed for '{self.model_name}'. Applied offline patches to {len(applied_offline_keys)} keys. Online layers: {len(self.online_lora_layers)}.")

    def unload(self):
        """Restores the model to its original state and clears backups."""
        model = self._get_model()
        if model is None: return

        logger.info(f"Unloading all LoRAs and restoring original state for '{self.model_name}'.")
        original_parameter_devices = get_parameter_devices(model) # Get current (potentially modified) devices

        # Clear online LoRAs first
        self._clear_online_loras(model)

        # Restore weights from any existing backup
        self._restore_weights(model, original_parameter_devices)

        # Ensure parameters end up on the correct devices
        set_parameter_devices(model, original_parameter_devices, force_move=True)

        # Clear internal state
        self.backup.clear()
        self.online_lora_layers.clear()
        self.loaded_lora_config_hash = None
        # Optionally clear the global cache if unloading suggests a major state change
        # clear_lora_cache()
        logger.info(f"LoRA unloading complete for '{self.model_name}'.")
        memory_management.soft_empty_cache()
