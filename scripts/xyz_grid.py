# Enhanced X/Y/Z Plot Script with Novel Features
# Original base: Likely from Automatic1111/Forge Web UI
# Enhancements: Presets, Python Expr, File Input, Runtime Est, Preview,
#             GIF Output, Custom API, Interpolation, Stubs for BestOf/External.
# WARNING: Highly complex, generated code. Requires testing & adaptation.

from collections import namedtuple
from copy import copy, deepcopy
from itertools import permutations, product, cycle
import random
import csv
import os
import os.path
import json
import re
import time
import math # Added for potential use in Python expressions
import traceback
from io import StringIO
from PIL import Image
import numpy as np
import asteval # Using asteval for safer evaluation of Python expressions

# Try importing imageio for GIF generation, make it optional
try:
    import imageio
    imageio_available = True
except ImportError:
    imageio_available = False
    print("XYZ Plot: imageio library not found. GIF generation disabled. Install with: pip install imageio")

import modules.scripts as scripts
import gradio as gr

from modules import images, sd_samplers, processing, sd_models, sd_vae, sd_schedulers, errors
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, state, prompt_styles
from modules.sd_models import model_data, select_checkpoint, checkpoints_list
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import modules.scripts # Added for potential external script interaction

from modules.ui_components import ToolButton

# --- Constants and Symbols ---
fill_values_symbol = "\U0001f4d2"  # 📒
save_symbol = "\U0001f4be"  # 💾
delete_symbol = "\U0001f5d1\ufe0f" # 🗑️
PRESET_DIR = os.path.join(scripts.basedir(), "xyz_presets") # Directory to store presets
os.makedirs(PRESET_DIR, exist_ok=True)

# --- Global Variables ---
# Registry for custom axis options added by other scripts
xyz_custom_axis_options = []

# --- Named Tuples ---
AxisInfo = namedtuple('AxisInfo', ['axis', 'values'])

# --- Helper Functions ---

def get_axis_opts():
    """Returns the list of available axis options, including custom ones."""
    # Filter base options based on txt2img/img2img context
    current_base_options = [x for x in axis_options_base if type(x) == AxisOption or x.is_img2img == shared.cmd_opts.img2img] # Use shared.cmd_opts as is_img2img isn't directly available here
    # Combine base and custom options
    return current_base_options + xyz_custom_axis_options

def list_to_csv_string(data_list):
    """Converts a list of items to a comma-separated string."""
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()

def csv_string_to_list_strip(data_str):
    """Converts a comma-separated string to a list of stripped strings."""
    # Handle potential empty strings or strings containing only whitespace
    if not data_str or data_str.isspace():
        return []
    try:
        return list(map(str.strip, chain.from_iterable(csv.reader(StringIO(data_str), skipinitialspace=True))))
    except (csv.Error, TypeError):
        # Fallback for simple non-CSV strings or errors
        return [data_str.strip()] if data_str else []

def get_current_script_options(p):
    """ Placeholder: Get parameters for other active scripts. Highly environment-dependent. """
    # This needs to be adapted based on how the specific Web UI stores script info
    # Example for A1111-style (might need adjustment):
    script_options = {}
    if hasattr(p, 'scripts') and p.scripts and hasattr(p.scripts, 'scripts'):
         for script in p.scripts.scripts:
             if hasattr(script, 'args_from') and hasattr(script, 'args_to'):
                script_title = script.title().lower().replace(" ", "_")
                script_args = p.script_args[script.args_from:script.args_to]
                # Try to find parameter names (this is fragile)
                try:
                    # Attempt to get param names from the script's ui method definition
                    ui_inputs = script.ui(script.is_img2img).inputs # This might fail or not exist
                    param_names = [getattr(comp, 'label', f'param_{i}') for i, comp in enumerate(ui_inputs)]
                    script_options[script_title] = dict(zip(param_names, script_args))
                except Exception:
                    # Fallback if param names can't be found
                     script_options[script_title] = {f'param_{i}': arg for i, arg in enumerate(script_args)}
    return script_options

# --- Preset Management ---

def get_preset_choices():
    """Returns a list of saved preset filenames."""
    return [f for f in os.listdir(PRESET_DIR) if f.endswith(".json")]

def load_preset(preset_filename):
    """Loads settings from a preset file."""
    if not preset_filename:
        return [gr.update()] * 17 # Return updates for all relevant UI elements
    filepath = os.path.join(PRESET_DIR, preset_filename)
    if not os.path.exists(filepath):
        print(f"XYZ Plot: Preset not found: {preset_filename}")
        return [gr.update()] * 17

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Need to map saved label back to index for dropdowns
        axis_labels = [opt.label for opt in get_axis_opts()]
        x_type_index = axis_labels.index(data.get("x_type", axis_options_base[1].label)) # Default Seed
        y_type_index = axis_labels.index(data.get("y_type", axis_options_base[0].label)) # Default Nothing
        z_type_index = axis_labels.index(data.get("z_type", axis_options_base[0].label)) # Default Nothing

        # Handle potential differences in available options between saves
        x_type_index = x_type_index if x_type_index < len(axis_labels) else 0
        y_type_index = y_type_index if y_type_index < len(axis_labels) else 0
        z_type_index = z_type_index if z_type_index < len(axis_labels) else 0

        # Return updates for all relevant components based on saved data
        return (
            gr.Dropdown.update(value=x_type_index),
            gr.Textbox.update(value=data.get("x_values", "")),
            gr.Dropdown.update(value=data.get("x_values_dropdown", [])), # Assuming dropdown values are saved if applicable
            gr.Dropdown.update(value=y_type_index),
            gr.Textbox.update(value=data.get("y_values", "")),
            gr.Dropdown.update(value=data.get("y_values_dropdown", [])),
            gr.Dropdown.update(value=z_type_index),
            gr.Textbox.update(value=data.get("z_values", "")),
            gr.Dropdown.update(value=data.get("z_values_dropdown", [])),
            gr.Checkbox.update(value=data.get("draw_legend", True)),
            gr.Checkbox.update(value=data.get("include_lone_images", False)),
            gr.Checkbox.update(value=data.get("include_sub_grids", False)),
            gr.Checkbox.update(value=data.get("no_fixed_seeds", False)),
            gr.Checkbox.update(value=data.get("vary_seeds_x", False)),
            gr.Checkbox.update(value=data.get("vary_seeds_y", False)),
            gr.Checkbox.update(value=data.get("vary_seeds_z", False)),
            gr.Slider.update(value=data.get("margin_size", 0)),
            gr.Checkbox.update(value=data.get("csv_mode", False)),
            # Add updates for GIF/other options if they were saved
            gr.Checkbox.update(value=data.get("gif_output_x", False)),
            gr.Checkbox.update(value=data.get("gif_output_y", False)),
            gr.Checkbox.update(value=data.get("gif_output_z", False)),
        )

    except Exception as e:
        print(f"XYZ Plot: Error loading preset '{preset_filename}': {e}")
        return [gr.update()] * 17 # Adjust count if more elements are returned

def save_preset(preset_filename, x_type_val, x_values, x_values_dropdown, y_type_val, y_values, y_values_dropdown, z_type_val, z_values, z_values_dropdown, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode, gif_x, gif_y, gif_z):
    """Saves the current settings to a preset file."""
    if not preset_filename:
        return gr.update(choices=get_preset_choices())

    # Sanitize filename
    preset_filename = "".join(c for c in preset_filename if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not preset_filename:
        print("XYZ Plot: Invalid preset filename.")
        return gr.update(choices=get_preset_choices())
    
    filepath = os.path.join(PRESET_DIR, f"{preset_filename}.json")

    # Get labels corresponding to selected indices
    axis_opts = get_axis_opts()
    x_label = axis_opts[x_type_val].label if x_type_val < len(axis_opts) else axis_options_base[1].label
    y_label = axis_opts[y_type_val].label if y_type_val < len(axis_opts) else axis_options_base[0].label
    z_label = axis_opts[z_type_val].label if z_type_val < len(axis_opts) else axis_options_base[0].label

    data = {
        "x_type": x_label,
        "x_values": x_values,
        "x_values_dropdown": x_values_dropdown,
        "y_type": y_label,
        "y_values": y_values,
        "y_values_dropdown": y_values_dropdown,
        "z_type": z_label,
        "z_values": z_values,
        "z_values_dropdown": z_values_dropdown,
        "draw_legend": draw_legend,
        "include_lone_images": include_lone_images,
        "include_sub_grids": include_sub_grids,
        "no_fixed_seeds": no_fixed_seeds,
        "vary_seeds_x": vary_seeds_x,
        "vary_seeds_y": vary_seeds_y,
        "vary_seeds_z": vary_seeds_z,
        "margin_size": margin_size,
        "csv_mode": csv_mode,
        "gif_output_x": gif_x,
        "gif_output_y": gif_y,
        "gif_output_z": gif_z,
        # Add other options as needed
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"XYZ Plot: Preset saved: {preset_filename}.json")
        return gr.update(choices=get_preset_choices(), value=f"{preset_filename}.json") # Update dropdown choices and select the new one
    except Exception as e:
        print(f"XYZ Plot: Error saving preset '{preset_filename}': {e}")
        return gr.update(choices=get_preset_choices())

def delete_preset(preset_filename):
    """Deletes a selected preset file."""
    if not preset_filename:
        return gr.update(choices=get_preset_choices())

    filepath = os.path.join(PRESET_DIR, preset_filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"XYZ Plot: Preset deleted: {preset_filename}")
            return gr.update(choices=get_preset_choices(), value=None) # Update choices and clear selection
        except Exception as e:
            print(f"XYZ Plot: Error deleting preset '{preset_filename}': {e}")
            return gr.update(choices=get_preset_choices())
    else:
        print(f"XYZ Plot: Preset not found for deletion: {preset_filename}")
        return gr.update(choices=get_preset_choices())


# --- Axis Option Application Functions ---
# (Keep existing apply_field, apply_prompt, apply_order, etc.)
# Add apply_external_script stub

def apply_field(field):
    def fun(p, x, xs):
        # Special handling for hr_sampler_name if the selected sampler is not valid for Img2Img
        # (This might be better handled in confirmation or processing logic)
        if field == 'hr_sampler_name' and x not in sd_samplers.samplers_for_img2img_map:
            print(f"Warning: Hires sampler '{x}' not typically used for img2img. Result may vary.")
            # Optionally, fallback to a default img2img sampler or raise error earlier
        setattr(p, field, x)
    return fun

def apply_prompt(p, x, xs):
    # Ensure xs[0] exists and is a string before searching
    if not xs or not isinstance(xs[0], str) or not xs[0]:
         print(f"Warning: Prompt S/R base string is empty or invalid. Skipping.")
         return
    # Handle cases where the search term might not be in prompt or negative prompt
    prompt_changed = False
    if xs[0] in p.prompt:
        p.prompt = p.prompt.replace(xs[0], x)
        prompt_changed = True
    if xs[0] in p.negative_prompt:
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)
        prompt_changed = True
        
    if not prompt_changed:
        # Optionally raise error or just print warning
        print(f"Warning: Prompt S/R did not find '{xs[0]}' in prompt or negative prompt.")
        # raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")


def apply_order(p, x, xs):
    # x should be a permutation (tuple or list of strings)
    if not isinstance(x, (list, tuple)) or not all(isinstance(s, str) for s in x):
        print(f"Error: Invalid value type for Prompt order: {type(x)}. Expected list/tuple of strings.")
        return # Or raise error

    token_order = []
    temp_prompt = p.prompt # Work on a copy to handle overlapping tokens correctly

    # Find occurrences and their start indices
    for token in x:
        start_index = temp_prompt.find(token)
        if start_index != -1:
            token_order.append({'token': token, 'index': start_index})
            # Mark the found token to avoid finding it again if tokens overlap or are substrings
            temp_prompt = temp_prompt[:start_index] + '#' * len(token) + temp_prompt[start_index + len(token):]
        else:
            # Handle case where a token is not found
            print(f"Warning: Token '{token}' not found in prompt for Prompt order.")
            # Decide whether to skip this permutation or raise error

    # Sort tokens by their original position
    token_order.sort(key=lambda t: t['index'])

    # Rebuild the prompt
    rebuilt_prompt = ""
    last_index = 0
    original_prompt_copy = p.prompt # Use the original prompt for slicing

    for item in token_order:
        # Add the text segment before the current token's original position
        rebuilt_prompt += original_prompt_copy[last_index:item['index']]
        # Add the token itself (from the permutation)
        rebuilt_prompt += item['token'] # Using item['token'] which is from the original 'x' permutation order
        # Update the position in the original prompt
        last_index = item['index'] + len(item['token'])

    # Add any remaining part of the original prompt after the last token
    rebuilt_prompt += original_prompt_copy[last_index:]

    p.prompt = rebuilt_prompt


def apply_checkpoint(p, x, xs):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    # Check if the checkpoint is already loaded or scheduled to be loaded
    # This comparison logic might need adjustment based on how overrides interact with the main opts
    current_model_name = p.override_settings.get('sd_model_checkpoint', getattr(opts, 'sd_model_checkpoint', None))
    # Use info.name for comparison as it's the canonical name
    if info.name == current_model_name:
         # print(f"Skipping reload for already active checkpoint: {info.name}")
         return
    # Apply override - the web UI's processing logic should handle the actual reload
    p.override_settings['sd_model_checkpoint'] = info.name
    # # Potentially force model parameter refresh if needed by the specific UI fork (Forge example)
    # if hasattr(modules.sd_models, 'model_data') and hasattr(modules.sd_models.model_data, 'forge_loading_parameters'):
    #     refresh_loading_params_for_xyz_grid() # If this function exists and is needed


def apply_vae(p, x, xs):
    vae_name = find_vae(x) # find_vae handles 'auto', 'none', and finds matches
    # Similar check to avoid unnecessary overrides if already set
    current_vae = p.override_settings.get('sd_vae', getattr(opts, 'sd_vae', None))
    if vae_name == current_vae:
        # print(f"Skipping VAE change, already set to: {vae_name}")
        return
    p.override_settings['sd_vae'] = vae_name


# --- Stubs/Placeholders for Complex Features ---

def apply_external_script_param(p, x, xs):
    """
    Stub for applying a parameter to another script.
    Requires significant adaptation for the target Web UI.
    Expected 'x' format: 'script_name_or_title;parameter_name_or_index;value'
    """
    try:
        script_target, param_target, value_str = x.split(';', 2)
        script_target = script_target.strip().lower().replace(" ", "_")
        param_target = param_target.strip()
        
        # Convert value_str to appropriate type (this is basic, might need more robust logic)
        try:
            value = float(value_str)
            if value.is_integer(): value = int(value)
        except ValueError:
            if value_str.lower() == 'true': value = True
            elif value_str.lower() == 'false': value = False
            else: value = value_str # Keep as string

        # --- THIS IS THE HIGHLY ENVIRONMENT-SPECIFIC PART ---
        # How to find the script and modify its arguments in 'p'?
        if hasattr(p, 'scripts') and p.scripts and hasattr(p.scripts, 'scripts'):
            found_script = None
            target_script_obj = None
            for script_obj in p.scripts.scripts:
                if script_obj.title().lower().replace(" ", "_") == script_target:
                    found_script = True
                    target_script_obj = script_obj
                    break
            
            if found_script and target_script_obj:
                 # Now, find the parameter index
                 param_index = -1
                 # Try matching by label/name first (requires script exposing this info)
                 try:
                    # Hypothetical way to get param names, WILL LIKELY FAIL
                    ui_inputs = target_script_obj.ui(target_script_obj.is_img2img).inputs 
                    param_names = [getattr(comp, 'label', f'param_{i}').lower().replace(" ", "_") for i, comp in enumerate(ui_inputs)]
                    if param_target.lower().replace(" ", "_") in param_names:
                        param_index = param_names.index(param_target.lower().replace(" ", "_"))
                 except Exception:
                     pass # Failed to get names, try index

                 # If not found by name, try direct index if param_target is numeric
                 if param_index == -1 and param_target.isdigit():
                     param_index = int(param_target)

                 # If we have a valid index relative to the script's args
                 if param_index != -1 and hasattr(target_script_obj, 'args_from') and hasattr(target_script_obj, 'args_to'):
                      actual_index = target_script_obj.args_from + param_index
                      if 0 <= actual_index < len(p.script_args):
                          # Modify the argument in the main processing object's script_args list
                          p.script_args[actual_index] = value
                          print(f"XYZ Plot: Applied to External Script '{script_target}': Param '{param_target}' (index {actual_index}) = {value}")
                          return # Success
                      else:
                           print(f"XYZ Plot Warning: Calculated index {actual_index} out of bounds for script '{script_target}'.")
                 else:
                     print(f"XYZ Plot Warning: Could not find parameter '{param_target}' for script '{script_target}' by name or index.")
            else:
                print(f"XYZ Plot Warning: External script '{script_target}' not found or not active.")

        else:
            print("XYZ Plot Warning: Cannot access script data structure 'p.scripts.scripts'. External Script axis may not work.")

    except Exception as e:
        print(f"XYZ Plot Error applying external script param '{x}': {e}")
        # traceback.print_exc() # Uncomment for debugging

# --- Interpolation Axis Handling ---
def parse_interpolation_string(val_str):
    """ Parses strings like 'param1=start1->end1; param2=start2->end2; steps=N' """
    parts = [p.strip() for p in val_str.split(';') if p.strip()]
    params = {}
    steps = 5 # Default steps

    for part in parts:
        if part.lower().startswith('steps='):
            try:
                steps = int(part.split('=')[1].strip())
                if steps < 2: steps = 2 # Need at least 2 steps for interpolation
            except ValueError:
                raise ValueError(f"Invalid steps value in interpolation string: {part}")
        else:
            match = re.match(r"(.+?)\s*=\s*(.+?)\s*->\s*(.+)", part)
            if match:
                name, start_str, end_str = match.groups()
                name = name.strip()
                try:
                    # Attempt to convert to float, fallback to string if needed (more robust type handling needed)
                    start_val = float(start_str.strip())
                    end_val = float(end_str.strip())
                    # Convert to int if they are integers
                    if start_val.is_integer(): start_val = int(start_val)
                    if end_val.is_integer(): end_val = int(end_val)
                    params[name] = {'start': start_val, 'end': end_val}
                except ValueError:
                     # Could treat as string interpolation if needed, but numeric is more common
                     raise ValueError(f"Non-numeric start/end value in interpolation string: {part}")
            else:
                 raise ValueError(f"Invalid parameter format in interpolation string: {part}")
                 
    if not params:
        raise ValueError("No valid parameters found in interpolation string.")
        
    return params, steps

def generate_interpolation_values(params, steps):
    """ Generates lists of interpolated values for each parameter """
    interpolated_values = {}
    for name, data in params.items():
        # Use numpy.linspace for smooth interpolation
        interpolated_values[name] = np.linspace(data['start'], data['end'], steps).tolist()
        # Round floats to reasonable precision
        if isinstance(data['start'], float) or isinstance(data['end'], float):
            interpolated_values[name] = [round(v, 6) for v in interpolated_values[name]]
        # Convert back to int if start/end were ints
        if isinstance(data['start'], int) and isinstance(data['end'], int):
            interpolated_values[name] = [int(round(v)) for v in interpolated_values[name]]
            
    # Structure: list of dictionaries, each dict is a step with values for all params
    num_params = len(params)
    structured_steps = []
    param_names = list(params.keys())
    
    for i in range(steps):
        step_data = {name: interpolated_values[name][i] for name in param_names}
        structured_steps.append(step_data)
        
    return structured_steps # Example: [{'cfg': 5, 'steps': 20}, {'cfg': 6.25, 'steps': 25}, ...]


def apply_interpolation(p, x, xs):
     """ Applies the interpolated values for a single step (x is a dict) """
     # Find the AxisOption objects corresponding to the parameter names in x
     axis_opts_map = {opt.label: opt for opt in get_axis_opts()}
     
     for param_name, value in x.items():
         if param_name in axis_opts_map:
             axis_opt = axis_opts_map[param_name]
             # Apply the value using the specific AxisOption's apply function
             # The 'xs' argument here is technically incorrect context for interpolation,
             # but most apply functions don't use it heavily. Pass an empty list or None.
             try:
                 axis_opt.apply(p, value, []) 
             except Exception as e:
                  print(f"Error applying interpolated value for {param_name}: {e}")
         else:
             print(f"Warning: Interpolation parameter '{param_name}' does not match any known AxisOption label.")

def format_interpolation_value(p, opt, x):
    """ Formats the dictionary of interpolated values for display """
    return "; ".join([f"{k}={v}" for k, v in x.items()])


# --- AxisOption Definition ---
# (Includes existing options and adds new ones/stubs)

class AxisOption:
    def __init__(self, label, type, apply, format_value=format_value_add_label, confirm=None, cost=0.0, choices=None, prepare=None, is_img2img=None):
        self.label = label
        self.type = type
        self.apply = apply
        self.format_value = format_value # How to display the value in legends/UI
        self.confirm = confirm # Function to validate values before run
        self.cost = cost # Heuristic cost for optimizing loop order (higher is slower)
        self.choices = choices # Function returning list of choices for dropdowns
        self.prepare = prepare # Function to process input value string before type conversion (rarely needed now with process_axis handling)
        self.is_img2img = is_img2img # None = works for both, True = img2img only, False = txt2img only


def confirm_samplers(p, xs):
    # Check against combined map for flexibility, but apply logic might warn later
    valid_samplers = sd_samplers.all_samplers_map
    for x in xs:
        if x.lower() not in valid_samplers:
            raise RuntimeError(f"Unknown sampler: {x}")


def confirm_checkpoints(p, xs):
    loaded_checkpoints = {info.name: info for info in checkpoints_list.values()}
    for x in xs:
        info = modules.sd_models.get_closet_checkpoint_match(x)
        if info is None or info.name not in loaded_checkpoints:
             # Check if it exists but isn't loaded (less common case)
             all_checkpoints = modules.sd_models.checkpoints_list
             if x in all_checkpoints or any(c.lower() == x.lower() for c in all_checkpoints):
                 print(f"Warning: Checkpoint '{x}' found but might not be fully loaded or matched correctly.")
             else:
                raise RuntimeError(f"Unknown or inaccessible checkpoint: {x}")


def find_vae(name: str):
    if not isinstance(name, str): name = str(name) # Ensure string input
    name_lower = name.strip().lower()
    if name_lower in ('auto', 'automatic'): return 'Automatic'
    if name_lower == 'none': return 'None'
    
    # Check loaded VAEs first
    if name in modules.sd_vae.vae_dict: return name
    if name_lower in modules.sd_vae.vae_dict_lower: return modules.sd_vae.vae_dict_lower[name_lower]

    # Fallback search (might be slow if many VAEs)
    vae_matches = [k for k in modules.sd_vae.vae_dict if k.lower() == name_lower]
    if vae_matches: return vae_matches[0]

    print(f'Warning: No VAE found matching "{name}". Using Automatic.')
    return 'Automatic'


def apply_styles(p: processing.StableDiffusionProcessing, x: str, _):
     # Styles are additive from the base P object, ensure we don't duplicate
     # Clears previous script-added styles and adds current ones
     base_styles = getattr(p, '_xyz_original_styles', p.styles) # Get original styles if saved, else current
     p._xyz_original_styles = base_styles # Save original styles on first application
     
     # Split potentially comma-separated style string from the axis value
     styles_to_apply = [s.strip() for s in x.split(',') if s.strip()]
     
     # Combine base styles and the new styles for this step
     # Use a set to avoid duplicates while preserving order as much as possible
     combined_styles = list(dict.fromkeys(base_styles + styles_to_apply))
     p.styles = combined_styles


def str_permutations(x):
    """Dummy function for type hint for permutations"""
    return x # Type is actually list/tuple of strings

def format_value_add_label(p, opt, x):
    if isinstance(x, float): x = round(x, 8)
    # Handle list/tuple for permutation display
    if isinstance(x, (list, tuple)): x = ", ".join(map(str, x))
    return f"{opt.label}: {x}"

def format_value(p, opt, x):
    if isinstance(x, float): x = round(x, 8)
    if isinstance(x, (list, tuple)): x = ", ".join(map(str, x))
    return x

def format_nothing(p, opt, x): return ""
def do_nothing(p, x, xs): pass
def format_remove_path(p, opt, x): return os.path.basename(x) if isinstance(x, str) else x

# --- Base Axis Options List ---
# More options added below
axis_options_base = [
    AxisOption("Nothing", type(None), do_nothing, format_value=format_nothing), # Type changed to NoneType
    AxisOption("Seed", int, apply_field("seed")),
    AxisOption("Var. seed", int, apply_field("subseed")),
    AxisOption("Var. strength", float, apply_field("subseed_strength")),
    AxisOption("Steps", int, apply_field("steps")),
    AxisOption("CFG Scale", float, apply_field("cfg_scale")),
    AxisOption("Prompt S/R", str, apply_prompt, format_value=format_value, prepare=lambda x: x.split('->', 1) if '->' in x else [x, x]), # Allow 'find->replace' syntax
    AxisOption("Prompt order", str_permutations, apply_order, format_value=lambda p,o,x: f"Order: {', '.join(x)}"),
    AxisOption("Sampler", str, apply_field("sampler_name"), format_value=format_value, confirm=confirm_samplers, choices=lambda: [x.name for x in sd_samplers.all_samplers if x.name not in opts.hide_samplers], is_img2img=None), # Changed to allow all samplers initially
    AxisOption("Checkpoint name", str, apply_checkpoint, format_value=format_remove_path, confirm=confirm_checkpoints, cost=1.0, choices=lambda: sorted(modules.sd_models.checkpoints_list.keys(), key=str.casefold)),
    AxisOption("Sigma Churn", float, apply_field("s_churn")),
    AxisOption("Sigma min", float, apply_field("s_tmin")),
    AxisOption("Sigma max", float, apply_field("s_tmax")),
    AxisOption("Sigma noise", float, apply_field("s_noise")),
    AxisOption("Eta", float, apply_field("eta")), # Noise multiplier eta (DDIM)
    AxisOption("Eta Noise Seed Delta (ENSD)", int, apply_override("eta_noise_seed_delta"), cost=0.1), # Option for ENSD
    AxisOption("Clip skip", int, apply_override('CLIP_stop_at_last_layers')),
    AxisOption("VAE", str, apply_vae, cost=0.7, choices=lambda: ['Automatic', 'None'] + list(sd_vae.vae_dict)),
    AxisOption("Styles", str, apply_styles, choices=lambda: list(prompt_styles.styles)),
    AxisOption("Width", int, apply_field("width"), cost=0.1),
    AxisOption("Height", int, apply_field("height"), cost=0.1),
    AxisOption("Highres. fix", bool, apply_field("enable_hr"), cost=0.1, choices=lambda: ["True", "False"]), # Basic boolean choice
    AxisOption("Hires steps", int, apply_field("hr_second_pass_steps"), is_img2img=False),
    AxisOption("Hires upscale", float, apply_field("hr_scale")),
    AxisOption("Hires upscaler", str, apply_field("hr_upscaler"), choices=lambda: [*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]]),
    AxisOption("Denoising", float, apply_field("denoising_strength")),
    AxisOption("Cond. Image Mask Weight", float, apply_field("inpainting_mask_weight"), is_img2img=True), # Only relevant for inpainting img2img
    AxisOption("Refiner Checkpoint", str, apply_field('refiner_checkpoint'), format_value=format_remove_path, confirm=confirm_checkpoints_or_none, cost=1.0, choices=lambda: ['None'] + sorted(modules.sd_models.checkpoints_list.keys(), key=str.casefold)),
    AxisOption("Refiner Switch At", float, apply_field('refiner_switch_at')),

    # --- Added Features & Stubs ---
    AxisOption(
        "Interpolate", 
        dict, # Type becomes dict holding {'param': value, ...} for the current step
        apply_interpolation, 
        format_value=format_interpolation_value,
        prepare=lambda x: generate_interpolation_values(*parse_interpolation_string(x)), # Prepare parses string and generates all steps
        cost=0.1 # Cost depends on interpolated params, estimate low average
    ),
     AxisOption(
        "External Script Param", 
        str, # Value is 'script;param;value' string
        apply_external_script_param, 
        format_value=lambda p,o,x: f"Ext: {x}",
        cost=0.05 # Assume low cost, actual cost is unknown
        # Cannot easily provide choices or confirmation here
    ),
     # Placeholder for BestOf - very simplified
     AxisOption(
        "BestOf Seed (Random)", # Indicates it's a placeholder using random choice
        int, # Input type is number of seeds to try
        apply_field("seed"), # It will ultimately set the seed
        format_value=lambda p,o,x: f"BestOf Seed (n={x}, chosen={getattr(p, '_xyz_chosen_seed', 'N/A')})",
        cost=0.01 # Cost is mainly the extra generations it implies
        # 'prepare' will handle generating random seeds and storing them for the run method
        # 'apply' will use the chosen seed for the specific cell
    ),
    # Add other existing options if they were missing
    AxisOption("Token merging ratio", float, apply_override('token_merging_ratio')),
    AxisOption("Token merging ratio high-res", float, apply_override('token_merging_ratio_hr')),
]

# Filter options based on txt2img/img2img where needed more carefully
def get_filtered_axis_opts(is_img2img):
    opts = axis_options_base + xyz_custom_axis_options
    return [opt for opt in opts if opt.is_img2img is None or opt.is_img2img == is_img2img]


# --- Custom Axis Registration ---
def register_axis_option(axis_option: AxisOption):
    """Allows other scripts to register custom AxisOptions."""
    if isinstance(axis_option, AxisOption):
        print(f"XYZ Plot: Registering custom axis option: {axis_option.label}")
        xyz_custom_axis_options.append(axis_option)
    else:
        print("XYZ Plot Error: Attempted to register invalid axis option.")

# --- Main Script Class ---

class Script(scripts.Script):
    def title(self):
        return "X/Y/Z plot Enhanced" # Updated title

    # Store current axis options based on context
    current_axis_options = []

    def ui(self, is_img2img):
        self.is_img2img = is_img2img # Store context
        self.current_axis_options = get_filtered_axis_opts(is_img2img)
        axis_labels = [x.label for x in self.current_axis_options]
        # Default to Seed/Nothing/Nothing if possible, otherwise use index 0
        default_x_index = axis_labels.index("Seed") if "Seed" in axis_labels else 1
        default_y_index = axis_labels.index("Nothing") if "Nothing" in axis_labels else 0
        default_z_index = axis_labels.index("Nothing") if "Nothing" in axis_labels else 0

        with gr.Blocks(): # Use Blocks for more layout control
             # Preset Management Section
            with gr.Accordion("Presets", open=False):
                 with gr.Row():
                     preset_dropdown = gr.Dropdown(label="Load Preset", choices=get_preset_choices(), elem_id=self.elem_id("xyz_preset_load"))
                     load_button = gr.Button("Load")
                     delete_button = ToolButton(value=delete_symbol, tooltip="Delete selected preset")
                 with gr.Row():
                     preset_save_name = gr.Textbox(label="Save Preset As", elem_id=self.elem_id("xyz_preset_save_name"))
                     save_button = ToolButton(value=save_symbol, tooltip="Save current XYZ settings")
                     
            # Axis Configuration Section
            with gr.Row():
                with gr.Column(scale=19):
                     # X Axis
                     with gr.Row():
                         x_type = gr.Dropdown(label="X type", choices=axis_labels, value=axis_labels[default_x_index] if default_x_index < len(axis_labels) else axis_labels[0], type="index", elem_id=self.elem_id("x_type"))
                         x_values_textbox = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"), placeholder="Comma-separated values or ranges (e.g., 1-5, 10-20[5], py:[x/2 for x in range(4)])")
                         x_values_dropdown = gr.Dropdown(label="X values (Dropdown)", visible=False, multiselect=True, interactive=True)
                         x_values_file = gr.File(label="X values (File)", file_count="single", file_types=[".txt", ".csv", ".json"], visible=True, elem_id=self.elem_id("x_file")) # File input always potentially visible
                         fill_x_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_x_tool_button", visible=False)
                     # Y Axis
                     with gr.Row():
                         y_type = gr.Dropdown(label="Y type", choices=axis_labels, value=axis_labels[default_y_index] if default_y_index < len(axis_labels) else axis_labels[0], type="index", elem_id=self.elem_id("y_type"))
                         y_values_textbox = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"), placeholder="Comma-separated values or ranges...")
                         y_values_dropdown = gr.Dropdown(label="Y values (Dropdown)", visible=False, multiselect=True, interactive=True)
                         y_values_file = gr.File(label="Y values (File)", file_count="single", file_types=[".txt", ".csv", ".json"], visible=True, elem_id=self.elem_id("y_file"))
                         fill_y_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_y_tool_button", visible=False)
                     # Z Axis
                     with gr.Row():
                         z_type = gr.Dropdown(label="Z type", choices=axis_labels, value=axis_labels[default_z_index] if default_z_index < len(axis_labels) else axis_labels[0], type="index", elem_id=self.elem_id("z_type"))
                         z_values_textbox = gr.Textbox(label="Z values", lines=1, elem_id=self.elem_id("z_values"), placeholder="Comma-separated values or ranges...")
                         z_values_dropdown = gr.Dropdown(label="Z values (Dropdown)", visible=False, multiselect=True, interactive=True)
                         z_values_file = gr.File(label="Z values (File)", file_count="single", file_types=[".txt", ".csv", ".json"], visible=True, elem_id=self.elem_id("z_file"))
                         fill_z_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_z_tool_button", visible=False)

            # Options Section
            with gr.Row(variant="compact", elem_id="axis_options"):
                 with gr.Column():
                     draw_legend = gr.Checkbox(label='Draw legend', value=True, elem_id=self.elem_id("draw_legend"))
                     no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))
                     with gr.Row():
                         vary_seeds_x = gr.Checkbox(label='Vary seeds for X', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_x"), tooltip="Use different seeds for images along X axis.")
                         vary_seeds_y = gr.Checkbox(label='Vary seeds for Y', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_y"), tooltip="Use different seeds for images along Y axis.")
                         vary_seeds_z = gr.Checkbox(label='Vary seeds for Z', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_z"), tooltip="Use different seeds for images along Z axis.")
                 with gr.Column():
                     include_lone_images = gr.Checkbox(label='Include Sub Images', value=False, elem_id=self.elem_id("include_lone_images"))
                     include_sub_grids = gr.Checkbox(label='Include Sub Grids', value=False, elem_id=self.elem_id("include_sub_grids"))
                     csv_mode = gr.Checkbox(label='Use text inputs instead of dropdowns', value=False, elem_id=self.elem_id("csv_mode")) # Keep this? Or rely on file input? Keeping for now.
                 with gr.Column():
                     margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))
                     # GIF Options
                     with gr.Row():
                          create_gif_x = gr.Checkbox(label='GIF for X-axis', value=False, elem_id=self.elem_id("gif_output_x"), visible=imageio_available)
                          create_gif_y = gr.Checkbox(label='GIF for Y-axis', value=False, elem_id=self.elem_id("gif_output_y"), visible=imageio_available)
                          create_gif_z = gr.Checkbox(label='GIF for Z-axis', value=False, elem_id=self.elem_id("gif_output_z"), visible=imageio_available)


            # Swap Axes Section
            with gr.Row(variant="compact", elem_id="swap_axes"):
                 swap_xy_axes_button = gr.Button(value="Swap X/Y axes", elem_id="xy_grid_swap_axes_button")
                 swap_yz_axes_button = gr.Button(value="Swap Y/Z axes", elem_id="yz_grid_swap_axes_button")
                 swap_xz_axes_button = gr.Button(value="Swap X/Z axes", elem_id="xz_grid_swap_axes_button")

            # Preview and Estimation Section
            with gr.Accordion("Preview & Estimation", open=False):
                 preview_button = gr.Button("Update Preview & Estimate")
                 grid_preview = gr.Textbox(label="Grid Structure Preview", lines=4, interactive=False)
                 runtime_estimation = gr.Textbox(label="Estimated Runtime", interactive=False)


            # --- UI Event Handlers ---

            # Swap Axes Logic
            def swap_axes(ax1_type, ax1_val_txt, ax1_val_drop, ax1_val_file, ax2_type, ax2_val_txt, ax2_val_drop, ax2_val_file):
                return ax2_type, ax2_val_txt, ax2_val_drop, ax2_val_file, ax1_type, ax1_val_txt, ax1_val_drop, ax1_val_file

            xy_swap_args = [x_type, x_values_textbox, x_values_dropdown, x_values_file, y_type, y_values_textbox, y_values_dropdown, y_values_file]
            swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
            yz_swap_args = [y_type, y_values_textbox, y_values_dropdown, y_values_file, z_type, z_values_textbox, z_values_dropdown, z_values_file]
            swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
            xz_swap_args = [x_type, x_values_textbox, x_values_dropdown, x_values_file, z_type, z_values_textbox, z_values_dropdown, z_values_file]
            swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

            # Fill Button Logic
            def fill(axis_type_idx, csv_mode_active):
                axis = self.current_axis_options[axis_type_idx]
                choices_func = axis.choices
                if choices_func:
                    choices_list = choices_func()
                    if csv_mode_active:
                        # Fill textbox
                        return gr.Textbox.update(value=list_to_csv_string(choices_list)), gr.Dropdown.update(value=[]), gr.File.update(value=None)
                    else:
                        # Fill dropdown
                        return gr.Textbox.update(value=""), gr.Dropdown.update(choices=choices_list, value=choices_list), gr.File.update(value=None) # Select all by default
                else:
                    # No choices available
                     return gr.Textbox.update(), gr.Dropdown.update(), gr.File.update() # No change

            fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values_textbox, x_values_dropdown, x_values_file])
            fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values_textbox, y_values_dropdown, y_values_file])
            fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values_textbox, z_values_dropdown, z_values_file])


            # Axis Type Change Logic (Handles visibility of Textbox/Dropdown)
            def select_axis(axis_type_idx, axis_val_txt, axis_val_drop, axis_val_file, csv_mode_active):
                # Ensure axis_type_idx is valid
                if axis_type_idx is None or axis_type_idx >= len(self.current_axis_options):
                     axis_type_idx = 0 # Default to first option if invalid
                
                axis = self.current_axis_options[axis_type_idx]
                choices_func = axis.choices
                has_choices = choices_func is not None

                # Determine visibility based on choices and mode
                show_dropdown = has_choices and not csv_mode_active
                show_textbox = not show_dropdown
                show_fill_button = has_choices # Show fill if choices exist, regardless of mode

                # Update dropdown choices if it's potentially visible
                dropdown_choices = choices_func() if has_choices else None
                
                # Logic to transfer values between textbox and dropdown when mode changes
                current_txt = axis_val_txt
                current_drop = axis_val_drop
                
                # If switching TO dropdown mode FROM text mode
                if show_dropdown and not axis_val_drop and axis_val_txt:
                    try:
                        current_drop = list(filter(lambda x: x in dropdown_choices, csv_string_to_list_strip(axis_val_txt)))
                        current_txt = "" # Clear textbox if transfer successful
                    except: pass # Ignore errors during conversion

                # If switching TO text mode FROM dropdown mode
                elif show_textbox and not axis_val_txt and axis_val_drop:
                     current_txt = list_to_csv_string(axis_val_drop)
                     current_drop = [] # Clear dropdown
                     
                # Clear file input if other inputs are used
                current_file = axis_val_file
                # if current_txt or current_drop:
                #     current_file = None # This causes issues if file is loaded then type changed

                return (
                    gr.Button.update(visible=show_fill_button),
                    gr.Textbox.update(visible=show_textbox, value=current_txt),
                    gr.Dropdown.update(visible=show_dropdown, choices=dropdown_choices, value=current_drop),
                    # gr.File.update(value=current_file) # Avoid clearing file automatically on type change
                )

            change_inputs = [x_type, x_values_textbox, x_values_dropdown, x_values_file, csv_mode]
            change_outputs = [fill_x_button, x_values_textbox, x_values_dropdown] # File output removed to avoid auto-clear
            x_type.change(fn=select_axis, inputs=change_inputs, outputs=change_outputs)

            change_inputs = [y_type, y_values_textbox, y_values_dropdown, y_values_file, csv_mode]
            change_outputs = [fill_y_button, y_values_textbox, y_values_dropdown]
            y_type.change(fn=select_axis, inputs=change_inputs, outputs=change_outputs)

            change_inputs = [z_type, z_values_textbox, z_values_dropdown, z_values_file, csv_mode]
            change_outputs = [fill_z_button, z_values_textbox, z_values_dropdown]
            z_type.change(fn=select_axis, inputs=change_inputs, outputs=change_outputs)


            # CSV Mode Change Logic (updates all axes)
            def change_choice_mode(csv_mode_active, xt, xv_txt, xv_drop, xv_file, yt, yv_txt, yv_drop, yv_file, zt, zv_txt, zv_drop, zv_file):
                fx, x_txt, x_drop = select_axis(xt, xv_txt, xv_drop, xv_file, csv_mode_active)
                fy, y_txt, y_drop = select_axis(yt, yv_txt, yv_drop, yv_file, csv_mode_active)
                fz, z_txt, z_drop = select_axis(zt, zv_txt, zv_drop, zv_file, csv_mode_active)
                return fx, x_txt, x_drop, fy, y_txt, y_drop, fz, z_txt, z_drop

            csv_mode_inputs = [csv_mode, x_type, x_values_textbox, x_values_dropdown, x_values_file, y_type, y_values_textbox, y_values_dropdown, y_values_file, z_type, z_values_textbox, z_values_dropdown, z_values_file]
            csv_mode_outputs= [fill_x_button, x_values_textbox, x_values_dropdown, fill_y_button, y_values_textbox, y_values_dropdown, fill_z_button, z_values_textbox, z_values_dropdown]
            csv_mode.change(fn=change_choice_mode, inputs=csv_mode_inputs, outputs=csv_mode_outputs)


            # Preview & Estimation Logic
            def update_preview(xt, xv_txt, xv_drop, xv_file, yt, yv_txt, yv_drop, yv_file, zt, zv_txt, zv_drop, zv_file, csv_mode_active):
                try:
                    # Use a dummy 'p' object for estimates, only needs basic fields
                    class DummyP:
                        n_iter = 1
                        batch_size = 1
                        steps = 20 # Assume default steps for cost calc if not varied
                        hr_second_pass_steps = 0 # Assume default
                        enable_hr = False
                        width = 512
                        height = 512

                    dummy_p = DummyP()

                    # Process axes values (similar to run method but without full processing)
                    x_opt = self.current_axis_options[xt]
                    y_opt = self.current_axis_options[yt]
                    z_opt = self.current_axis_options[zt]
                    
                    # Simplified value fetching (ignoring dropdown logic complexity for preview)
                    # Prioritize file, then textbox
                    xv = self._get_values_from_input(xv_txt, xv_drop, xv_file, csv_mode_active, x_opt)
                    yv = self._get_values_from_input(yv_txt, yv_drop, yv_file, csv_mode_active, y_opt)
                    zv = self._get_values_from_input(zv_txt, zv_drop, zv_file, csv_mode_active, z_opt)

                    xs = self._process_axis_values(x_opt, xv, csv_mode_active, dummy_p)
                    ys = self._process_axis_values(y_opt, yv, csv_mode_active, dummy_p)
                    zs = self._process_axis_values(z_opt, zv, csv_mode_active, dummy_p)

                    num_x, num_y, num_z = len(xs), len(ys), len(zs)
                    total_images = num_x * num_y * num_z

                    # Preview Text
                    preview_text = f"Grid Dimensions:\nX: {num_x} ({x_opt.label})\nY: {num_y} ({y_opt.label})\nZ: {num_z} ({z_opt.label})\n\nTotal Images: {total_images}"
                    
                    # Abbreviated Values (optional, can be long)
                    # preview_text += f"\n\nX Values: {str(xs[:3]) + ('...' if num_x > 3 else '')}"
                    # preview_text += f"\nY Values: {str(ys[:3]) + ('...' if num_y > 3 else '')}"
                    # preview_text += f"\nZ Values: {str(zs[:3]) + ('...' if num_z > 3 else '')}"


                    # Runtime Estimation (Very Rough)
                    cost_per_image_sec = 5.0 # BASE ESTIMATE: seconds per image (adjust!)
                    checkpoint_load_sec = 15.0 # Estimate for loading checkpoint
                    vae_load_sec = 5.0 # Estimate for loading VAE
                    
                    # Account for variable steps/hires
                    total_steps_estimate = self._estimate_total_steps(dummy_p, x_opt, xs, y_opt, ys, z_opt, zs)
                    base_runtime = (total_steps_estimate / dummy_p.steps) * cost_per_image_sec if dummy_p.steps > 0 else total_images * cost_per_image_sec

                    # Add costs based on loop order optimization heuristic
                    costs = {'x': x_opt.cost, 'y': y_opt.cost, 'z': z_opt.cost}
                    sorted_axes = sorted(costs, key=costs.get) # Slowest axis last
                    
                    # Calculate how many times slow operations run
                    loads_z = 1 if num_z > 1 else 0 # Z runs once per Z value (outermost)
                    loads_y = num_z if num_y > 1 else 0 # Y runs once per Z*Y combo
                    loads_x = num_z * num_y if num_x > 1 else 0 # X runs once per Z*Y*X combo

                    total_cost = base_runtime
                    
                    if z_opt.cost >= 1.0 and num_z > 1: total_cost += loads_z * checkpoint_load_sec
                    elif z_opt.cost >= 0.7 and num_z > 1: total_cost += loads_z * vae_load_sec
                    
                    if y_opt.cost >= 1.0 and num_y > 1: total_cost += loads_y * checkpoint_load_sec
                    elif y_opt.cost >= 0.7 and num_y > 1: total_cost += loads_y * vae_load_sec
                    
                    if x_opt.cost >= 1.0 and num_x > 1: total_cost += loads_x * checkpoint_load_sec
                    elif x_opt.cost >= 0.7 and num_x > 1: total_cost += loads_x * vae_load_sec
                    
                    # Format runtime
                    if total_cost < 60:
                        est_runtime_str = f"~ {total_cost:.1f} seconds"
                    elif total_cost < 3600:
                         est_runtime_str = f"~ {total_cost / 60:.1f} minutes"
                    else:
                         est_runtime_str = f"~ {total_cost / 3600:.1f} hours"

                    return preview_text, est_runtime_str

                except Exception as e:
                    # traceback.print_exc() # Uncomment for debug
                    return f"Error generating preview: {e}", "Estimation failed"

            preview_inputs = [x_type, x_values_textbox, x_values_dropdown, x_values_file,
                              y_type, y_values_textbox, y_values_dropdown, y_values_file,
                              z_type, z_values_textbox, z_values_dropdown, z_values_file, csv_mode]
            preview_button.click(fn=update_preview, inputs=preview_inputs, outputs=[grid_preview, runtime_estimation])


            # Preset Buttons Handlers
            load_button.click(fn=load_preset,
                              inputs=[preset_dropdown],
                              outputs=[x_type, x_values_textbox, x_values_dropdown,
                                       y_type, y_values_textbox, y_values_dropdown,
                                       z_type, z_values_textbox, z_values_dropdown,
                                       draw_legend, include_lone_images, include_sub_grids,
                                       no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z,
                                       margin_size, csv_mode, create_gif_x, create_gif_y, create_gif_z]) # Ensure all controllable UI elements are outputs
            
            save_button.click(fn=save_preset,
                               inputs=[preset_save_name, x_type, x_values_textbox, x_values_dropdown,
                                       y_type, y_values_textbox, y_values_dropdown,
                                       z_type, z_values_textbox, z_values_dropdown,
                                       draw_legend, include_lone_images, include_sub_grids,
                                       no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z,
                                       margin_size, csv_mode, create_gif_x, create_gif_y, create_gif_z], # Pass all UI elements to save
                               outputs=[preset_dropdown]) # Update the preset list
            
            delete_button.click(fn=delete_preset, inputs=[preset_dropdown], outputs=[preset_dropdown])


            # Define infotext fields for loading parameters from PNG info
            self.infotext_fields = [
                (x_type, "X Type"), (x_values_textbox, "X Values"),
                (y_type, "Y Type"), (y_values_textbox, "Y Values"),
                (z_type, "Z Type"), (z_values_textbox, "Z Values"),
                # Add dropdowns if you want to attempt loading them (requires parsing CSV string back)
                # (x_values_dropdown, lambda p: gr.update(value=csv_string_to_list_strip(p.get("X Values","")))),
                # (y_values_dropdown, lambda p: gr.update(value=csv_string_to_list_strip(p.get("Y Values","")))),
                # (z_values_dropdown, lambda p: gr.update(value=csv_string_to_list_strip(p.get("Z Values","")))),
            ]
            # Also add checkboxes etc. if they should be loaded from infotext
            self.infotext_fields.extend([
                (draw_legend, "Draw legend"),
                (include_lone_images, "Include Sub Images"),
                (include_sub_grids, "Include Sub Grids"),
                (no_fixed_seeds, "Keep -1 for seeds"),
                (vary_seeds_x, "Vary seeds for X"),
                (vary_seeds_y, "Vary seeds for Y"),
                (vary_seeds_z, "Vary seeds for Z"),
                (margin_size, "Grid margins (px)"),
                (csv_mode, "Use text inputs instead of dropdowns"),
                 # Add GIF options if needed
            ])

        # The final list of components returned to Gradio for the script's run method
        return [preset_dropdown, preset_save_name, # Preset management UI elements (might not be needed in run args)
                x_type, x_values_textbox, x_values_dropdown, x_values_file,
                y_type, y_values_textbox, y_values_dropdown, y_values_file,
                z_type, z_values_textbox, z_values_dropdown, z_values_file,
                draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds,
                vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode,
                create_gif_x, create_gif_y, create_gif_z] # Add new options


    def _get_values_from_input(self, values_txt, values_drop, values_file, csv_mode_active, axis_opt):
        """Helper to get values prioritizing File > Textbox > Dropdown."""
        if values_file is not None:
            # Read from file
            try:
                filepath = values_file.name # Gradio File object has 'name' attribute for temp path
                if filepath.lower().endswith(".csv"):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        # Assuming values are in the first column
                        return [row[0].strip() for row in reader if row]
                elif filepath.lower().endswith(".json"):
                     with open(filepath, 'r', encoding='utf-8') as f:
                         data = json.load(f)
                         if isinstance(data, list):
                             return [str(item) for item in data] # Assume list of values
                         else: raise ValueError("JSON file must contain a list of values.")
                else: # Assume TXT file, one value per line
                    with open(filepath, 'r', encoding='utf-8') as f:
                        return [line.strip() for line in f if line.strip()]
            except Exception as e:
                raise ValueError(f"Error reading file {os.path.basename(values_file.name)}: {e}")
        elif axis_opt.choices is not None and not csv_mode_active:
            # Use dropdown values if applicable and not in CSV mode
            return values_drop
        else:
            # Use textbox values otherwise
            return values_txt # This will be parsed later

    def _process_axis_values(self, opt: AxisOption, vals_input, csv_mode_active, p: processing.StableDiffusionProcessing):
        """Processes the input value string/list/file into a list of values of the correct type."""
        
        if opt.label == 'Nothing':
            return [None] # Use None for the single value of "Nothing" axis

        # vals_input can be: string (from textbox), list (from dropdown/file), list of dicts (interpolation prepare)
        
        # Handle pre-prepared values (like interpolation)
        if opt.prepare and isinstance(vals_input, str): # Only call prepare if input is the raw string
             # The prepare function (e.g., for interpolation) returns the final list of values
             valslist = opt.prepare(vals_input)
             # Interpolation prepare returns list of dicts, other prepares might return lists of strings
             if isinstance(valslist, list) and valslist and isinstance(valslist[0], dict): # Check if it looks like interpolation output
                 return valslist # Return the list of dicts directly
             # Otherwise, assume prepare returned a list of strings to be processed further
             valslist_processed = valslist
        elif isinstance(vals_input, list):
             valslist_processed = [str(v) for v in vals_input] # Ensure all are strings for parsing
        elif isinstance(vals_input, str):
            # Check for Python expression
            if vals_input.strip().startswith("py:"):
                py_expr = vals_input.strip()[3:]
                try:
                    # Use asteval for safer evaluation
                    aeval = asteval.Interpreter()
                    # Disable potentially harmful functions if needed (asteval is relatively safe)
                    # aeval.no_deepcopy = True 
                    # aeval.no_print = True
                    # Add safe modules/functions
                    aeval.symtable['math'] = math
                    aeval.symtable['np'] = np # Allow numpy if needed
                    aeval.symtable['random'] = random 
                    
                    result = aeval.eval(py_expr)
                    if not isinstance(result, list):
                        raise TypeError("Python expression must return a list.")
                    valslist_processed = [str(x) for x in result] # Convert results to string for uniform processing below
                except Exception as e:
                    raise ValueError(f"Error evaluating Python expression: {e}\nExpression: {py_expr}")
            else:
                # Parse CSV string
                valslist_processed = csv_string_to_list_strip(vals_input)
        else:
             # Handle None or unexpected input types
             valslist_processed = []

        # Now, valslist_processed should be a list of strings. Parse ranges and convert types.
        valslist_final = []
        if not valslist_processed and opt.label != 'Nothing':
            # Handle empty input for required axes (e.g., raise error or use a default?)
            # Let's allow empty values for flexibility, apply function might handle it.
            pass
            # Example: For checkpoint, maybe default to current?
            # if opt.label == 'Checkpoint name': return [getattr(opts, 'sd_model_checkpoint', '')]

        # Range parsing logic (integer and float)
        re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]?\d+)\s*\))?\s*") # Step syntax: (...)
        re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*])?\s*") # Count syntax: [...]
        re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]?\d+(?:.\d*)?)\s*\))?\s*")
        re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+)\s*])?\s*") # Use int for count

        for val_str in valslist_processed:
            val_str = val_str.strip()
            if not val_str and opt.type is not str: # Allow empty strings if type is string
                 continue

            # Integer Range Expansion
            if opt.type == int:
                m = re_range.fullmatch(val_str)
                mc = re_range_count.fullmatch(val_str)
                if m:
                    start, end = int(m.group(1)), int(m.group(2))
                    step = int(m.group(3)) if m.group(3) else 1
                    if step == 0: raise ValueError("Step cannot be zero.")
                    # Adjust end point for range/arange behavior
                    # Python range excludes end, numpy arange includes it depending on step
                    valslist_final.extend(list(range(start, end + (1 if step > 0 else -1), step)))
                elif mc:
                    start, end = int(mc.group(1)), int(mc.group(2))
                    num = int(mc.group(3)) if mc.group(3) else 2 # Default to 2 if count omitted
                    if num <= 0: raise ValueError("Count must be positive.")
                    if num == 1: valslist_final.append(start) # Special case for single point
                    else: valslist_final.extend(np.linspace(start=start, stop=end, num=num, dtype=int).tolist())
                else:
                    try: valslist_final.append(int(val_str))
                    except ValueError: raise ValueError(f"Invalid integer value: '{val_str}' for axis '{opt.label}'")
            # Float Range Expansion
            elif opt.type == float:
                m = re_range_float.fullmatch(val_str)
                mc = re_range_count_float.fullmatch(val_str)
                if m:
                    start, end = float(m.group(1)), float(m.group(2))
                    step = float(m.group(3)) if m.group(3) else 1.0
                    if step == 0: raise ValueError("Step cannot be zero.")
                    # Use np.arange for float steps, being mindful of precision issues
                    valslist_final.extend(np.arange(start, end + step * 0.5 * np.sign(step), step).tolist()) # Add small epsilon to include end
                elif mc:
                    start, end = float(mc.group(1)), float(mc.group(2))
                    num = int(mc.group(3)) if mc.group(3) else 2
                    if num <= 0: raise ValueError("Count must be positive.")
                    if num == 1: valslist_final.append(start)
                    else: valslist_final.extend(np.linspace(start=start, stop=end, num=num).tolist())
                else:
                    try: valslist_final.append(float(val_str))
                    except ValueError: raise ValueError(f"Invalid float value: '{val_str}' for axis '{opt.label}'")
            # String Permutations
            elif opt.type == str_permutations:
                 # Values are collected first, then permutations are generated *after* all values are parsed.
                 # So here, just add the string. The permutation happens later.
                 valslist_final.append(val_str)
            # Boolean Type
            elif opt.type == bool:
                 valslist_final.append(val_str.lower() in ('true', '1', 'yes', 'y'))
            # Default: Treat as String or direct type conversion
            else:
                try:
                    # Directly convert if type is simple (e.g., str) or add raw string otherwise
                    if opt.type == str:
                        valslist_final.append(val_str)
                    else:
                        valslist_final.append(opt.type(val_str)) # Attempt direct conversion for other types if needed
                except (ValueError, TypeError):
                    # Fallback to string if conversion fails
                    print(f"Warning: Could not convert '{val_str}' to type {opt.type.__name__} for axis '{opt.label}'. Using as string.")
                    valslist_final.append(val_str)


        # Handle Permutations specifically *after* parsing all strings
        if opt.type == str_permutations:
             if not valslist_final: return []
             # Generate all permutations of the collected strings
             valslist_final = list(permutations(valslist_final))

        # Final type conversion pass (mostly for float rounding)
        processed_values = []
        if valslist_final:
             val_type = opt.type
             # Handle special types that produce lists/dicts
             if val_type == str_permutations: val_type = list # Permutations are lists
             if opt.label == "Interpolate": val_type = dict # Interpolation steps are dicts
             
             for val in valslist_final:
                 try:
                     # Apply final type conversion or rounding
                     if val_type == float:
                         processed_values.append(round(float(val), 8))
                     elif val_type == int:
                          processed_values.append(int(val))
                     elif val_type == bool:
                          processed_values.append(bool(val))
                     elif val_type == list or val_type == dict: # Handle permutations/interpolation
                         processed_values.append(val)
                     else: # Default to string or original type
                         processed_values.append(str(val) if val_type == str else val) 
                 except (ValueError, TypeError) as e:
                      raise ValueError(f"Failed final type conversion for value '{val}' to {val_type.__name__} on axis '{opt.label}': {e}")


        # Confirm options are valid before returning
        if opt.confirm:
            opt.confirm(p, processed_values)

        # Handle "BestOf Seed" placeholder: input is count, output is list of that many -1s
        if opt.label.startswith("BestOf Seed"):
            if not processed_values or not isinstance(processed_values[0], int) or processed_values[0] <= 0:
                raise ValueError("BestOf Seed axis requires a positive integer count as input.")
            num_seeds_to_try = processed_values[0]
            # We return a list representing the *iterations*, not the seeds themselves yet.
            # The 'run' method will handle generating random seeds for these iterations.
            return list(range(num_seeds_to_try)) # Example: input 5 -> output [0, 1, 2, 3, 4]

        return processed_values


    def _estimate_total_steps(self, p, x_opt, xs, y_opt, ys, z_opt, zs):
        """Estimates total steps based on axis types and hires settings."""
        num_x, num_y, num_z = len(xs), len(ys), len(zs)
        total_cells = num_x * num_y * num_z
        
        # Base steps
        base_steps_per_cell = p.steps
        if x_opt.label == 'Steps': total_base_steps = sum(xs) * num_y * num_z
        elif y_opt.label == 'Steps': total_base_steps = sum(ys) * num_x * num_z
        elif z_opt.label == 'Steps': total_base_steps = sum(zs) * num_x * num_y
        else: total_base_steps = base_steps_per_cell * total_cells

        # Hires steps (if txt2img and enabled)
        total_hr_steps = 0
        if isinstance(p, StableDiffusionProcessingTxt2Img) and p.enable_hr:
            hr_steps_per_cell = p.hr_second_pass_steps or p.steps # Use base steps if hr steps is 0
            if x_opt.label == "Hires steps": total_hr_steps = sum(xs) * num_y * num_z
            elif y_opt.label == "Hires steps": total_hr_steps = sum(ys) * num_x * num_z
            elif z_opt.label == "Hires steps": total_hr_steps = sum(zs) * num_x * num_y
            elif x_opt.label == "Highres. fix" and any(v for v in xs): # If HR fix is enabled by X axis
                 # Average step count (rough estimate)
                 total_hr_steps = hr_steps_per_cell * sum(1 for v in xs if v) * num_y * num_z
            elif y_opt.label == "Highres. fix" and any(v for v in ys): # If HR fix is enabled by Y axis
                 total_hr_steps = hr_steps_per_cell * sum(1 for v in ys if v) * num_x * num_z
            elif z_opt.label == "Highres. fix" and any(v for v in zs): # If HR fix is enabled by Z axis
                 total_hr_steps = hr_steps_per_cell * sum(1 for v in zs if v) * num_x * num_y
            else: # HR enabled but not varied by axes
                total_hr_steps = hr_steps_per_cell * total_cells
                
        total_steps = (total_base_steps + total_hr_steps) * p.n_iter * p.batch_size
        return total_steps


    def run(self, p: processing.StableDiffusionProcessing,
            preset_dropdown_arg_ignore, preset_save_name_arg_ignore, # Ignore preset UI elements in run
            x_type_idx, x_values_textbox, x_values_dropdown, x_values_file,
            y_type_idx, y_values_textbox, y_values_dropdown, y_values_file,
            z_type_idx, z_values_textbox, z_values_dropdown, z_values_file,
            draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds,
            vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode,
            create_gif_x, create_gif_y, create_gif_z # Added GIF options
           ):

        # --- Input Processing and Validation ---
        start_time = time.time()

        # Get current axis options based on run context (p is StableDiffusionProcessing object)
        self.is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
        self.current_axis_options = get_filtered_axis_opts(self.is_img2img)

        # Validate indices
        x_type_idx = x_type_idx if 0 <= x_type_idx < len(self.current_axis_options) else 0
        y_type_idx = y_type_idx if 0 <= y_type_idx < len(self.current_axis_options) else 0
        z_type_idx = z_type_idx if 0 <= z_type_idx < len(self.current_axis_options) else 0

        x_opt = self.current_axis_options[x_type_idx]
        y_opt = self.current_axis_options[y_type_idx]
        z_opt = self.current_axis_options[z_type_idx]

        # Get values using helper function
        try:
            xv = self._get_values_from_input(x_values_textbox, x_values_dropdown, x_values_file, csv_mode, x_opt)
            yv = self._get_values_from_input(y_values_textbox, y_values_dropdown, y_values_file, csv_mode, y_opt)
            zv = self._get_values_from_input(z_values_textbox, z_values_dropdown, z_values_file, csv_mode, z_opt)
        except ValueError as e:
             errors.report(f"XYZ Plot: Error reading input values: {e}", exc_info=True)
             return Processed(p, [], p.seed, "") # Return empty processed on error

        # Process values (parse ranges, types, etc.)
        try:
            xs_raw = self._process_axis_values(x_opt, xv, csv_mode, p)
            ys_raw = self._process_axis_values(y_opt, yv, csv_mode, p)
            zs_raw = self._process_axis_values(z_opt, zv, csv_mode, p)
        except (ValueError, TypeError) as e:
            errors.report(f"XYZ Plot: Error processing axis values: {e}", exc_info=True)
            # traceback.print_exc() # Uncomment for debugging console
            return Processed(p, [], p.seed, "")

        # --- Handle "BestOf Seed" Placeholder ---
        # If an axis is BestOf, we need multiple runs per cell.
        # The _process_axis_values returned a list of indices [0, 1, ..., n-1] for BestOf.
        bestof_axis = None
        bestof_n = 1
        if x_opt.label.startswith("BestOf Seed"):
            bestof_axis = 'x'
            bestof_n = len(xs_raw) if xs_raw else 1
            xs = [0] # We only need one value for the main loop, cell fn will handle iterations
            x_labels_processed = [f"BestOf Seed (n={bestof_n})"] # Single label for the combined axis
        else:
            xs = xs_raw
            x_labels_processed = [x_opt.format_value(p, x_opt, x) for x in xs]

        if y_opt.label.startswith("BestOf Seed"):
            if bestof_axis: raise ValueError("Only one axis can be 'BestOf Seed'")
            bestof_axis = 'y'
            bestof_n = len(ys_raw) if ys_raw else 1
            ys = [0]
            y_labels_processed = [f"BestOf Seed (n={bestof_n})"]
        else:
            ys = ys_raw
            y_labels_processed = [y_opt.format_value(p, y_opt, y) for y in ys]

        if z_opt.label.startswith("BestOf Seed"):
            if bestof_axis: raise ValueError("Only one axis can be 'BestOf Seed'")
            bestof_axis = 'z'
            bestof_n = len(zs_raw) if zs_raw else 1
            zs = [0]
            z_labels_processed = [f"BestOf Seed (n={bestof_n})"]
        else:
            zs = zs_raw
            z_labels_processed = [z_opt.format_value(p, z_opt, z) for z in zs]
            
        if not xs or not ys or not zs:
            errors.report("XYZ Plot: At least one axis has no values.", exc_info=False)
            return Processed(p, [], p.seed, "")

        # Fix seeds if required
        if not no_fixed_seeds:
            modules.processing.fix_seed(p) # Ensure p.seed is fixed if -1

            # The fix_axis_seeds logic needs refinement as it relied on the old direct processing
            # If Seed axis exists and uses -1, it should be randomized *once* unless vary_seeds is on.
            # We'll handle the randomization inside the 'cell' function if needed.
            pass # Initial randomization handled by fix_seed(p)

        # --- Grid Size & Step Calculation ---
        num_x, num_y, num_z = len(xs), len(ys), len(zs)
        total_images_nominal = num_x * num_y * num_z * p.n_iter * p.batch_size * bestof_n # Include BestOf iterations
        
        # Estimate total steps (more accurate calculation)
        total_steps = self._estimate_total_steps(p, x_opt, xs_raw, y_opt, ys_raw, z_opt, zs_raw) # Use raw values for step estimation
        total_steps *= bestof_n # Multiply by number of BestOf runs

        print(f"XYZ Plot: Generating {total_images_nominal} images across {num_z} grids ({num_x}x{num_y} cells). Total steps: {total_steps}. BestOf: {bestof_axis} (n={bestof_n})" if bestof_axis else f"XYZ Plot: Generating {total_images_nominal} images across {num_z} grids ({num_x}x{num_y} cells). Total steps: {total_steps}.")
        
        # Check grid size limit BEFORE processing
        # Pillow check removal needs careful consideration of memory. Let's use the existing check.
        # Image.MAX_IMAGE_PIXELS = None # Disable check in Pillow? Risky.
        grid_mp = round(num_x * num_y * num_z * p.width * p.height / 1_000_000)
        if grid_mp > opts.img_max_size_mp:
             errors.report(f'XYZ Plot Error: Resulting grid would be too large ({grid_mp} MPixels). Max configured size is {opts.img_max_size_mp} MPixels. Reduce axis values or image dimensions.', exc_info=False)
             return Processed(p, [], p.seed, "")
        
        shared.total_tqdm.updateTotal(total_steps)
        
        # --- State and Loop Optimization ---
        state.job_count = num_x * num_y * num_z * p.n_iter * bestof_n # Total cells to process
        state.xyz_plot_x = AxisInfo(x_opt, xs_raw) # Store raw values for potential reference
        state.xyz_plot_y = AxisInfo(y_opt, ys_raw)
        state.xyz_plot_z = AxisInfo(z_opt, zs_raw)

        # Loop order optimization based on cost
        costs = {'x': x_opt.cost, 'y': y_opt.cost, 'z': z_opt.cost}
        # Give BestOf axis highest effective cost if present
        if bestof_axis == 'x': costs['x'] += 100
        if bestof_axis == 'y': costs['y'] += 100
        if bestof_axis == 'z': costs['z'] += 100
        
        sorted_axes = sorted(costs, key=costs.get) # Process cheapest first (innermost loop)
        loop_order = sorted_axes[::-1] # Outermost loop is most expensive
        first_axis, second_axis, third_axis = loop_order

        print(f"XYZ Plot: Loop order: {first_axis} (outer) -> {second_axis} -> {third_axis} (inner)")

        # --- Image Generation Loop ---
        # Store results temporarily
        # grid_images will store the final image for each cell after potential BestOf selection
        grid_images = {} # {(ix, iy, iz): image}
        # cell_results will store all generated images and info for a cell, needed for BestOf
        cell_results = {} # {(ix, iy, iz): {'images': [], 'infos': [], 'seeds': [], 'prompts': []}}

        original_prompt = p.prompt
        original_neg_prompt = p.negative_prompt
        original_seed = p.seed
        original_subseed = p.subseed
        original_styles = p.styles[:] # Keep a copy of initial styles

        processed_result_template = None # To store structure of Processed object

        # Create the nested loops dynamically based on optimized order
        axes_map = {'x': (xs, x_opt), 'y': (ys, y_opt), 'z': (zs, z_opt)}
        iters_map = {'x': range(num_x), 'y': range(num_y), 'z': range(num_z)}

        vals1, opt1 = axes_map[first_axis]
        vals2, opt2 = axes_map[second_axis]
        vals3, opt3 = axes_map[third_axis]
        
        iter1, iter2, iter3 = iters_map[first_axis], iters_map[second_axis], iters_map[third_axis]
        
        job_index = 0

        with SharedSettingsStackHelper(): # Manages model/VAE reloading if necessary
            for i1_idx in iter1:
                val1 = vals1[i1_idx]
                for i2_idx in iter2:
                    val2 = vals2[i2_idx]
                    for i3_idx in iter3:
                        val3 = vals3[i3_idx]

                        if state.interrupted or state.stopping_generation: break
                        
                        # Determine grid coordinates (ix, iy, iz) based on loop variables
                        coords = {}
                        coords[first_axis] = i1_idx
                        coords[second_axis] = i2_idx
                        coords[third_axis] = i3_idx
                        ix, iy, iz = coords['x'], coords['y'], coords['z']
                        
                        # Reset p object to original state for each cell, applying overrides later
                        pc = copy(p) # Shallow copy is usually enough, but styles need deep?
                        pc.prompt = original_prompt
                        pc.negative_prompt = original_neg_prompt
                        pc.seed = original_seed
                        pc.subseed = original_subseed
                        pc.styles = original_styles[:] # Reset styles
                        pc.override_settings = {} # Start fresh overrides
                        pc.cached_uc = {} # Clear cached cond/uncond
                        pc.cached_c = {}
                        
                        # --- Apply Axis Values ---
                        # Apply values based on the original (non-optimized) axis definitions (x, y, z)
                        
                        # Use the correct value depending on whether this axis is 'BestOf'
                        x_val_apply = xs_raw[ix] if bestof_axis != 'x' else -1 # Use raw value or placeholder
                        y_val_apply = ys_raw[iy] if bestof_axis != 'y' else -1
                        z_val_apply = zs_raw[iz] if bestof_axis != 'z' else -1
                        
                        if x_opt.label != "Nothing": x_opt.apply(pc, x_val_apply, xs_raw)
                        if y_opt.label != "Nothing": y_opt.apply(pc, y_val_apply, ys_raw)
                        if z_opt.label != "Nothing": z_opt.apply(pc, z_val_apply, zs_raw)

                        # --- BestOf Seed Logic ---
                        num_iterations_for_cell = bestof_n
                        cell_images_temp = []
                        cell_infos_temp = []
                        cell_seeds_temp = []
                        cell_prompts_temp = []

                        for bestof_iter in range(num_iterations_for_cell):
                            if state.interrupted or state.stopping_generation: break
                            
                            job_index += 1
                            state.job = f"{job_index} / {state.job_count}"
                            
                            # Make a copy of the cell's base 'pc' for this iteration
                            p_iter = deepcopy(pc) if bestof_axis else pc # Deepcopy if BestOf modifies p_iter significantly
                            
                            # Handle Seed Variation and BestOf Seed Setting
                            current_seed = p_iter.seed
                            seed_offset = 0
                            if vary_seeds_x: seed_offset += ix
                            if vary_seeds_y: seed_offset += iy * (num_x if vary_seeds_x else 1)
                            if vary_seeds_z: seed_offset += iz * (num_x if vary_seeds_x else 1) * (num_y if vary_seeds_y else 1)
                            
                            # If this cell is part of a BestOf axis run
                            if bestof_axis:
                                # Generate a truly random seed for each iteration of BestOf
                                iter_seed = int(random.randrange(4294967294))
                                p_iter.seed = iter_seed
                                # Store the chosen seed for potential display/metadata
                                p_iter._xyz_chosen_seed = iter_seed 
                                # Apply this specific seed using the relevant axis option's apply method
                                if bestof_axis == 'x': x_opt.apply(p_iter, iter_seed, [])
                                elif bestof_axis == 'y': y_opt.apply(p_iter, iter_seed, [])
                                elif bestof_axis == 'z': z_opt.apply(p_iter, iter_seed, [])
                            else:
                                # Apply standard seed variation if not BestOf
                                if not no_fixed_seeds and seed_offset > 0:
                                    p_iter.seed += seed_offset
                                elif not no_fixed_seeds and p_iter.seed == -1:
                                    p_iter.seed = int(random.randrange(4294967294)) # Randomize if -1 and not varied
                            
                            # Final check for -1 seed (should be randomized now unless no_fixed_seeds)
                            if p_iter.seed == -1 and not no_fixed_seeds:
                                 p_iter.seed = int(random.randrange(4294967294))

                            # Ensure subseed follows similar logic if needed
                            if p_iter.subseed == -1: p_iter.subseed = int(random.randrange(4294967294))

                            # --- Process Image(s) for this iteration ---
                            try:
                                # Clear cached conditions before each process_images call
                                p_iter.cached_uc = {}
                                p_iter.cached_c = {}

                                # Crucial: Ensure overrides are correctly handled by process_images
                                # This depends heavily on the Web UI's implementation.
                                # We assume process_images respects p_iter.override_settings
                                
                                processed_iter: Processed = process_images(p_iter)

                                if processed_result_template is None and processed_iter.images:
                                     # Copy the first valid Processed object to use as a template
                                     processed_result_template = copy(processed_iter)
                                     # Clear images and info from template
                                     processed_result_template.images = []
                                     processed_result_template.infotexts = []
                                     processed_result_template.all_prompts = []
                                     processed_result_template.all_seeds = []
                                     processed_result_template.all_subseeds = []
                                     processed_result_template.index_of_first_image = 0 # Will be adjusted later

                                if processed_iter.images:
                                    # Store results for this iteration (take first image if batch > 1)
                                    cell_images_temp.append(processed_iter.images[0])
                                    cell_infos_temp.append(processed_iter.infotexts[0])
                                    cell_seeds_temp.append(processed_iter.seed) # Or all_seeds?
                                    cell_prompts_temp.append(processed_iter.prompt) # Or all_prompts?
                                else:
                                    # Handle error case - append placeholder?
                                    print(f"Warning: Cell ({ix},{iy},{iz}), Iter {bestof_iter+1} failed to generate image.")
                                    # Append None or a blank image placeholder if needed for alignment
                                    cell_images_temp.append(None) 
                                    cell_infos_temp.append("")
                                    cell_seeds_temp.append(p_iter.seed)
                                    cell_prompts_temp.append(p_iter.prompt)


                            except Exception as e:
                                errors.report(f"Error generating image for cell ({ix},{iy},{iz}), iter {bestof_iter+1}: {e}", exc_info=True)
                                # Append None or placeholder on error
                                cell_images_temp.append(None)
                                cell_infos_temp.append(f"Error: {e}")
                                cell_seeds_temp.append(p_iter.seed)
                                cell_prompts_temp.append(p_iter.prompt)
                                # Optionally break the inner loop on error? Or continue?

                        # --- End of BestOf Iterations for Cell ---
                        if state.interrupted or state.stopping_generation: break
                        
                        # Store all results for the cell
                        cell_results[(ix, iy, iz)] = {
                            'images': cell_images_temp,
                            'infos': cell_infos_temp,
                            'seeds': cell_seeds_temp,
                            'prompts': cell_prompts_temp
                        }

                        # --- Select Best Image for the Cell (if BestOf) ---
                        if bestof_axis and cell_images_temp:
                             # Placeholder: Select randomly from generated images for this cell
                             valid_indices = [i for i, img in enumerate(cell_images_temp) if img is not None]
                             if valid_indices:
                                 chosen_index = random.choice(valid_indices)
                                 grid_images[(ix, iy, iz)] = cell_images_temp[chosen_index]
                                 # Store chosen seed in the info (or add dedicated field)
                                 chosen_info = cell_infos_temp[chosen_index]
                                 chosen_seed = cell_seeds_temp[chosen_index]
                                 # Add BestOf info to the infotext
                                 extra_info = f", BestOf Seed Choice: {chosen_seed} (out of {bestof_n})"
                                 # Find where to insert safely (e.g., before Steps)
                                 steps_match = re.search(r", Steps: \d+", chosen_info)
                                 if steps_match:
                                     insert_pos = steps_match.start()
                                     chosen_info = chosen_info[:insert_pos] + extra_info + chosen_info[insert_pos:]
                                 else:
                                      chosen_info += extra_info
                                 
                                 # Replace the list of results with the chosen one for this cell
                                 cell_results[(ix, iy, iz)] = {
                                     'images': [cell_images_temp[chosen_index]],
                                     'infos': [chosen_info],
                                     'seeds': [chosen_seed],
                                     'prompts': [cell_prompts_temp[chosen_index]]
                                 }

                             else:
                                 grid_images[(ix, iy, iz)] = None # No valid image generated
                                 cell_results[(ix, iy, iz)] = {'images': [None], 'infos': ["Error: No valid images in BestOf"], 'seeds': [-1], 'prompts': [""]}
                        elif cell_images_temp:
                             # If not BestOf, just take the first (and only) result
                             grid_images[(ix, iy, iz)] = cell_images_temp[0]
                        else:
                             grid_images[(ix, iy, iz)] = None # No image generated


                    if state.interrupted or state.stopping_generation: break
                if state.interrupted or state.stopping_generation: break
            if state.interrupted or state.stopping_generation: break
        
        # --- Post-Processing: Grid Assembly & Output ---

        if state.interrupted:
            print("XYZ Plot Generation interrupted.")
            # Fallback: return whatever was generated? Or empty?
            # Let's try returning what we have, it might be useful.
            
        if not processed_result_template:
             errors.report("XYZ Plot: No images were generated. Cannot create grid.", exc_info=False)
             return Processed(p, [], p.seed, "")

        # Create the final Processed object
        final_processed = processed_result_template
        final_processed.images = []
        final_processed.all_prompts = []
        final_processed.all_seeds = []
        final_processed.all_subseeds = [] # Subseeds not explicitly tracked here yet
        final_processed.infotexts = []
        final_processed.width = p.width # Use final p dimensions
        final_processed.height = p.height

        # Determine size of placeholder for failed cells
        cell_mode = "RGB"
        cell_size = (p.width, p.height)
        try:
            # Find first non-None image to get mode/size
            first_valid_image = next(img for img in grid_images.values() if img is not None)
            cell_mode = first_valid_image.mode
            cell_size = first_valid_image.size
        except StopIteration:
             print("Warning: No valid images found to determine grid cell mode/size. Using defaults.")

        # Populate the grid images list in Z, Y, X order for image_grid function
        flat_grid_images = []
        infotext_map = {} # Store infotext for saving grids later {(iz): grid_info, (ix,iy,iz): cell_info}
        
        # Generate extra params text for grids
        def create_grid_infotext(base_p, x_opt, x_vals_str, xs_proc, y_opt, y_vals_str, ys_proc, z_opt=None, z_vals_str=None, zs_proc=None, is_main_grid=False):
             pc_copy = copy(base_p) # Copy the initial p object
             pc_copy.extra_generation_params = copy(pc_copy.extra_generation_params) if hasattr(pc_copy, 'extra_generation_params') else {}
             pc_copy.extra_generation_params['Script'] = self.title()
             
             # Add axis info
             if x_opt.label != 'Nothing':
                 pc_copy.extra_generation_params["X Type"] = x_opt.label
                 pc_copy.extra_generation_params["X Values"] = x_vals_str # Original input string
                 if x_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                     pc_copy.extra_generation_params["Fixed X Values"] = ", ".join([str(x) for x in xs_proc])
             if y_opt.label != 'Nothing':
                 pc_copy.extra_generation_params["Y Type"] = y_opt.label
                 pc_copy.extra_generation_params["Y Values"] = y_vals_str
                 if y_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                     pc_copy.extra_generation_params["Fixed Y Values"] = ", ".join([str(y) for y in ys_proc])
             if z_opt and z_opt.label != 'Nothing':
                 pc_copy.extra_generation_params["Z Type"] = z_opt.label
                 pc_copy.extra_generation_params["Z Values"] = z_vals_str
                 if z_opt.label in ["Seed", "Var. seed"] and not no_fixed_seeds:
                      pc_copy.extra_generation_params["Fixed Z Values"] = ", ".join([str(z) for z in zs_proc])
             
             # Use the first cell's info as base for grid infotext (might lack some overrides)
             first_cell_info = "Grid Information" # Placeholder
             first_cell_coords = (0, 0, 0)
             if first_cell_coords in cell_results and cell_results[first_cell_coords]['infos']:
                  first_cell_info = cell_results[first_cell_coords]['infos'][0]
                  
             # Try to merge extra params with existing info string (basic approach)
             extra_params_str = ", ".join([f"{k}: {v}" for k, v in pc_copy.extra_generation_params.items()])
             
             # Simple append might duplicate info, needs smarter merging if possible
             merged_info = first_cell_info + ", " + extra_params_str
             
             # Alternative: Use processing.create_infotext with modified pc_copy
             # This might be cleaner if create_infotext handles extra_generation_params well
             try:
                 # Need placeholder prompts/seeds for create_infotext
                 all_prompts = [cell_results[first_cell_coords]['prompts'][0]] if first_cell_coords in cell_results and cell_results[first_cell_coords]['prompts'] else [pc_copy.prompt]
                 all_seeds = [cell_results[first_cell_coords]['seeds'][0]] if first_cell_coords in cell_results and cell_results[first_cell_coords]['seeds'] else [pc_copy.seed]
                 all_subseeds = [pc_copy.subseed] # Placeholder
                 
                 merged_info = processing.create_infotext(pc_copy, all_prompts, all_seeds, all_subseeds)
             except Exception as info_err:
                  print(f"Warning: Failed to use create_infotext for grid metadata: {info_err}")
                  # Fallback to simple merge
                  merged_info = first_cell_info + ", " + extra_params_str

             return merged_info


        # Get original input strings/lists for infotext
        x_vals_input_str = x_values_textbox if csv_mode or not x_opt.choices else list_to_csv_string(x_values_dropdown)
        y_vals_input_str = y_values_textbox if csv_mode or not y_opt.choices else list_to_csv_string(y_values_dropdown)
        z_vals_input_str = z_values_textbox if csv_mode or not z_opt.choices else list_to_csv_string(z_values_dropdown)
        # Add file source info if used
        if x_values_file: x_vals_input_str = f"File: {os.path.basename(x_values_file.name)}"
        if y_values_file: y_vals_input_str = f"File: {os.path.basename(y_values_file.name)}"
        if z_values_file: z_vals_input_str = f"File: {os.path.basename(z_values_file.name)}"


        # Generate main grid infotext (using Z axis info)
        main_grid_infotext = create_grid_infotext(p, x_opt, x_vals_input_str, xs_raw, y_opt, y_vals_input_str, ys_raw, z_opt, z_vals_input_str, zs_raw, is_main_grid=True)
        infotext_map[(-1)] = main_grid_infotext # Use -1 index for main grid info

        sub_grids = []
        for iz in range(num_z):
             grid_slice_images = []
             # Generate subgrid infotext
             sub_grid_infotext = create_grid_infotext(p, x_opt, x_vals_input_str, xs_raw, y_opt, y_vals_input_str, ys_raw) # Z info is implicitly the current iz
             infotext_map[(iz)] = sub_grid_infotext # Store subgrid info by iz
             
             for iy in range(num_y):
                 for ix in range(num_x):
                     img = grid_images.get((ix, iy, iz))
                     if img is None:
                         img = Image.new(cell_mode, cell_size, color="black") # Placeholder for failed images
                         
                     # Add individual image to final list if requested
                     if include_lone_images:
                         cell_info = cell_results.get((ix,iy,iz), {})
                         final_processed.images.append(img)
                         final_processed.infotexts.append(cell_info.get('infos', [""])[0])
                         final_processed.all_prompts.append(cell_info.get('prompts', [""])[0])
                         final_processed.all_seeds.append(cell_info.get('seeds', [-1])[0])
                         # Store cell info for potential saving
                         infotext_map[(ix, iy, iz)] = cell_info.get('infos', [""])[0]
                         
                     grid_slice_images.append(img)

             # Create subgrid image (Y rows, X columns)
             if num_x > 0 and num_y > 0:
                 subgrid_img = images.image_grid(grid_slice_images, rows=num_y)
                 if draw_legend:
                      # Use processed labels here
                      hor_texts = [[images.GridAnnotation(x)] for x in x_labels_processed]
                      ver_texts = [[images.GridAnnotation(y)] for y in y_labels_processed]
                      # Calculate max size within this slice for annotation
                      grid_max_w, grid_max_h = map(max, zip(*(img.size for img in grid_slice_images)))
                      subgrid_img = images.draw_grid_annotations(subgrid_img, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
                 sub_grids.append(subgrid_img)
                 
                 # Add subgrid to results if requested
                 if include_sub_grids:
                      final_processed.images.append(subgrid_img)
                      final_processed.infotexts.append(sub_grid_infotext)
                      # Use first cell's prompt/seed for subgrid representation
                      first_cell_info = cell_results.get((0,0,iz), {})
                      final_processed.all_prompts.append(first_cell_info.get('prompts', [""])[0])
                      final_processed.all_seeds.append(first_cell_info.get('seeds', [-1])[0])

        # Create main grid (Z columns, 1 row)
        if len(sub_grids) > 0 :
             main_grid_img = images.image_grid(sub_grids, rows=1)
             if draw_legend:
                 # Use processed Z labels
                 title_texts = [[images.GridAnnotation(z)] for z in z_labels_processed]
                 # Calculate max size across subgrids for annotation
                 grid_max_w, grid_max_h = map(max, zip(*(img.size for img in sub_grids)))
                 main_grid_img = images.draw_grid_annotations(main_grid_img, grid_max_w, grid_max_h, title_texts, [[images.GridAnnotation()]], margin_size) # Only Z titles

             # Add main grid to the beginning of the results
             final_processed.images.insert(0, main_grid_img)
             final_processed.infotexts.insert(0, main_grid_infotext)
             # Use first cell's prompt/seed for main grid representation
             first_cell_info = cell_results.get((0,0,0), {})
             final_processed.all_prompts.insert(0, first_cell_info.get('prompts', [""])[0])
             final_processed.all_seeds.insert(0, first_cell_info.get('seeds', [-1])[0])


        # --- GIF Generation ---
        if imageio_available and (create_gif_x or create_gif_y or create_gif_z):
            print("XYZ Plot: Generating GIFs...")
            gif_save_path = os.path.join(p.outpath_grids, "xyz_gifs")
            os.makedirs(gif_save_path, exist_ok=True)
            gif_duration = 0.5 # seconds per frame

            # Get all individual cell images (even if include_lone_images is false)
            all_cell_images = {}
            for iz in range(num_z):
                for iy in range(num_y):
                    for ix in range(num_x):
                         img = grid_images.get((ix, iy, iz))
                         if img: all_cell_images[(ix, iy, iz)] = img
                         else: all_cell_images[(ix, iy, iz)] = Image.new(cell_mode, cell_size, color="black") # Placeholder

            # Generate GIF cycling through X for each (Y, Z)
            if create_gif_x and num_x > 1:
                for iz in range(num_z):
                    for iy in range(num_y):
                         frames = [all_cell_images[(ix, iy, iz)] for ix in range(num_x)]
                         z_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(z_labels_processed[iz]))[:30]
                         y_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(y_labels_processed[iy]))[:30]
                         filename = f"x_axis_y_{y_label_safe}_z_{z_label_safe}.gif"
                         filepath = os.path.join(gif_save_path, filename)
                         imageio.mimsave(filepath, frames, duration=gif_duration, loop=0)
                         print(f"Saved X-axis GIF: {filename}")

            # Generate GIF cycling through Y for each (X, Z)
            if create_gif_y and num_y > 1:
                 for iz in range(num_z):
                    for ix in range(num_x):
                         frames = [all_cell_images[(ix, iy, iz)] for iy in range(num_y)]
                         z_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(z_labels_processed[iz]))[:30]
                         x_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(x_labels_processed[ix]))[:30]
                         filename = f"y_axis_x_{x_label_safe}_z_{z_label_safe}.gif"
                         filepath = os.path.join(gif_save_path, filename)
                         imageio.mimsave(filepath, frames, duration=gif_duration, loop=0)
                         print(f"Saved Y-axis GIF: {filename}")
                         
            # Generate GIF cycling through Z for each (X, Y)
            if create_gif_z and num_z > 1:
                 for iy in range(num_y):
                    for ix in range(num_x):
                         frames = [all_cell_images[(ix, iy, iz)] for iz in range(num_z)]
                         y_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(y_labels_processed[iy]))[:30]
                         x_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(x_labels_processed[ix]))[:30]
                         filename = f"z_axis_x_{x_label_safe}_y_{y_label_safe}.gif"
                         filepath = os.path.join(gif_save_path, filename)
                         imageio.mimsave(filepath, frames, duration=gif_duration, loop=0)
                         print(f"Saved Z-axis GIF: {filename}")


        # --- Final Cleanup & Saving ---
        # Auto-save grids if enabled
        if opts.grid_save:
            grid_count = 0
            # Save main grid
            if final_processed.images:
                 images.save_image(final_processed.images[0], p.outpath_grids, "xyz_grid_main", info=final_processed.infotexts[0], extension=opts.grid_format, prompt=final_processed.all_prompts[0], seed=final_processed.all_seeds[0], grid=True, p=p)
                 grid_count += 1
                 
            # Save subgrids if included and exist
            if include_sub_grids:
                 start_index = 1 # Subgrids start after main grid
                 end_index = start_index + num_z # There are num_z subgrids
                 if end_index <= len(final_processed.images):
                     for i in range(start_index, end_index):
                          subgrid_index = i - start_index # iz = 0, 1, ...
                          # Retrieve the correct infotext, prompt, seed for this subgrid
                          info = infotext_map.get((subgrid_index), "") # Get subgrid info by iz
                          prompt = final_processed.all_prompts[i] # Corresponding prompt/seed
                          seed = final_processed.all_seeds[i]
                          z_label_safe = re.sub(r'[\\/*?:"<>|]', '-', str(z_labels_processed[subgrid_index]))[:30]
                          images.save_image(final_processed.images[i], p.outpath_grids, f"xyz_grid_sub_z_{subgrid_index}_{z_label_safe}", info=info, extension=opts.grid_format, prompt=prompt, seed=seed, grid=True, p=p)
                          grid_count += 1
                          
        # Remove sub-images if not requested
        if not include_lone_images:
             # Find the index where lone images start
             lone_image_start_index = 1 # After main grid
             if include_sub_grids: lone_image_start_index += num_z # After sub grids
             
             if lone_image_start_index < len(final_processed.images):
                 del final_processed.images[lone_image_start_index:]
                 del final_processed.infotexts[lone_image_start_index:]
                 del final_processed.all_prompts[lone_image_start_index:]
                 del final_processed.all_seeds[lone_image_start_index:]
                 # Also remove from all_subseeds if tracked

        # Remove sub-grids if not requested (and they weren't removed by lone_images logic)
        elif not include_sub_grids: # This case only happens if include_lone_images=True but include_sub_grids=False
            subgrid_start_index = 1 # After main grid
            subgrid_end_index = subgrid_start_index + num_z
            if subgrid_end_index <= len(final_processed.images): # Check if subgrids actually exist in the list
                 del final_processed.images[subgrid_start_index:subgrid_end_index]
                 del final_processed.infotexts[subgrid_start_index:subgrid_end_index]
                 del final_processed.all_prompts[subgrid_start_index:subgrid_end_index]
                 del final_processed.all_seeds[subgrid_start_index:subgrid_end_index]
                 # Also remove from all_subseeds if tracked

        end_time = time.time()
        print(f"XYZ Plot Finished. Total time: {end_time - start_time:.2f} seconds.")

        return final_processed


# --- SharedSettingsStackHelper Class (for managing model/VAE reloads) ---
# (Keep existing implementation)
class SharedSettingsStackHelper(object):
    def __init__(self):
        self.checkpoint = None
        self.vae = None

    def __enter__(self):
        # Store original settings that might be changed and require reload
        self.checkpoint = opts.sd_model_checkpoint
        self.vae = opts.sd_vae
        # Add other settings if needed

    def __exit__(self, exc_type, exc_value, tb):
        # Restore original settings
        # Check if the setting actually changed before triggering reload
        # Note: Checkpoint/VAE might be handled by override_settings now,
        # making this less critical unless direct opts manipulation occurs.
        
        reload_model = False
        if self.checkpoint != opts.sd_model_checkpoint:
             opts.set('sd_model_checkpoint', self.checkpoint) # Restore setting in opts
             # Reloading might happen automatically via override_settings handling in main code,
             # or might need explicit trigger depending on UI version.
             # reload_model = True 
             print("XYZ Plot: Restoring original checkpoint setting.")

        reload_vae_flag = False
        if self.vae != opts.sd_vae:
             opts.set('sd_vae', self.vae)
             # reload_vae_flag = True
             print("XYZ Plot: Restoring original VAE setting.")

        # Explicitly reload if needed (may double-reload if overrides also trigger it)
        # if reload_model:
        #     modules.sd_models.reload_model_weights()
        # if reload_vae_flag:
        #     modules.sd_vae.reload_vae_weights()
        
        # If using Forge-style parameter refresh, call it here if necessary
        # if hasattr(modules.sd_models, 'model_data') and hasattr(modules.sd_models.model_data, 'forge_loading_parameters'):
        #     refresh_loading_params_for_xyz_grid() 


# Function to potentially refresh Forge loading parameters (if needed)
# Define it even if empty to avoid NameErrors if called
def refresh_loading_params_for_xyz_grid():
     """ Placeholder: Refreshes loading parameters for Forge UI if needed. """
     # This function's content depends entirely on the Forge Web UI's specific implementation
     # for handling model parameter changes during script execution.
     # If not using Forge or if not needed, leave this empty.
     # Example structure (adapt!):
     # if hasattr(modules.sd_models, 'model_data') and hasattr(modules.sd_models.model_data, 'forge_loading_parameters'):
     #     checkpoint_info = select_checkpoint() # Get currently selected checkpoint info
     #     # Update the parameters dictionary
     #     modules.sd_models.model_data.forge_loading_parameters = dict(
     #         checkpoint_info=checkpoint_info,
     #         # ... other parameters from shared.opts or model_data ...
     #     )
     #     print("XYZ Plot: Refreshed Forge loading parameters.")
     pass # Empty placeholder


# --- Utility functions (like confirm_checkpoints_or_none, apply_override etc) ---
# Keep existing or adapt as needed

def apply_override(field, boolean: bool = False, numeric_type=None):
    """Applies a value to p.override_settings, optionally converting type."""
    def fun(p, x, xs):
        value = x
        if boolean:
            value = str(x).lower() in ('true', '1', 'yes', 'y')
        elif numeric_type:
             try: value = numeric_type(x)
             except (ValueError, TypeError):
                 print(f"Warning: Could not convert override value '{x}' to {numeric_type.__name__} for field '{field}'. Keeping original.")
                 value = x # Keep original on error
        
        if not hasattr(p, 'override_settings'): p.override_settings = {}
        p.override_settings[field] = value
    return fun

def confirm_checkpoints_or_none(p, xs):
    loaded_checkpoints = {info.name: info for info in checkpoints_list.values()}
    for x in xs:
        if str(x).strip().lower() in ('none', ''): continue # Allow None/empty string

        info = modules.sd_models.get_closet_checkpoint_match(x)
        if info is None or info.name not in loaded_checkpoints:
            all_checkpoints = modules.sd_models.checkpoints_list
            if x in all_checkpoints or any(c.lower() == x.lower() for c in all_checkpoints):
                 print(f"Warning: Checkpoint '{x}' found but might not be fully loaded or matched correctly.")
            else:
                raise RuntimeError(f"Unknown or inaccessible checkpoint: {x}")

# Example boolean choice function
def boolean_choice(reverse: bool = False):
    return lambda: ["False", "True"] if reverse else ["True", "False"]
