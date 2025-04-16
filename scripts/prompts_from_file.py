import copy
import random
import shlex
import re # Added for variables and sweeping
import json # Added for saving job list
import os # Added for path manipulation
import math # Added for ceiling in grid calculation

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, errors, sd_models
from modules.processing import Processed, process_images, create_infotext
from modules.shared import state, opts
from modules.images import image_grid, save_image
from modules.safe import BannedPrompt

# --- Helper Functions ---

def sanitize_filename_part(text):
    """Removes invalid characters for filenames."""
    if not text:
        return ""
    # Remove Unicode characters that might cause issues, keep alphanumeric, -, _
    text = re.sub(r'[^\w\-_\. ]', '_', text, flags=re.UNICODE)
    # Remove leading/trailing spaces/dots
    text = text.strip(' .')
    # Limit length to avoid issues
    return text[:100]

def replace_placeholders(pattern: str, job_info: dict, p_obj, line_num: int = -1):
    """Replaces placeholders in a filename or path pattern."""
    if not pattern:
        return ""

    # Basic prompt slug (first few words)
    prompt_slug = ""
    if job_info.get('prompt'):
        prompt_slug = sanitize_filename_part(" ".join(job_info['prompt'].split()[:6]))

    # Full prompt slug (potentially long)
    full_prompt_slug = sanitize_filename_part(job_info.get('prompt', ''))

    # Negative prompt slug
    neg_prompt_slug = ""
    if job_info.get('negative_prompt'):
        neg_prompt_slug = sanitize_filename_part(" ".join(job_info['negative_prompt'].split()[:6]))

    # Sampler name
    sampler_name = job_info.get('sampler_name', p_obj.sampler_name)
    if hasattr(sampler_name, 'name'): # Handle sampler object vs name string
         sampler_name = sampler_name.name

    # Model name (get from override or original p)
    model_name = "DefaultModel"
    checkpoint_info = sd_models.get_closet_checkpoint_match(job_info.get('sd_model', p_obj.override_settings.get('sd_model_checkpoint', '')))
    if checkpoint_info:
        model_name = sanitize_filename_part(checkpoint_info.model_name) # Use model_name which is usually cleaner

    replacements = {
        "{prompt}": full_prompt_slug,
        "{prompt_slug}": prompt_slug,
        "{neg_prompt_slug}": neg_prompt_slug,
        "{seed}": str(job_info.get('seed', p_obj.seed)),
        "{subseed}": str(job_info.get('subseed', p_obj.subseed)),
        "{steps}": str(job_info.get('steps', p_obj.steps)),
        "{cfg_scale}": str(job_info.get('cfg_scale', p_obj.cfg_scale)),
        "{width}": str(job_info.get('width', p_obj.width)),
        "{height}": str(job_info.get('height', p_obj.height)),
        "{sampler_name}": sanitize_filename_part(sampler_name),
        "{model_name}": model_name, # Use sanitized model name
        "{line_num}": str(line_num) if line_num >= 0 else "",
        # Add more placeholders as needed
    }

    for key, value in replacements.items():
        pattern = pattern.replace(key, str(value)) # Ensure value is string

    # Sanitize the final pattern after replacements
    # Split into directory and filename parts to sanitize filename correctly
    dirname, basename = os.path.split(pattern)
    sanitized_basename = sanitize_filename_part(basename) # Sanitize only the filename part
    # Ensure directory part doesn't contain invalid chars (might be less strict than filename)
    sanitized_dirname = dirname # Basic sanitization could be added here if needed

    return os.path.join(sanitized_dirname, sanitized_basename) if dirname else sanitized_basename


# --- Parameter Parsing Logic ---

# Define processing functions for different types
def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    if info is None:
         errors.report(f"Warning: Unknown checkpoint specified: {tag}. Using default.", exc_info=False)
         return None # Signal to keep default
    return info.name # Return the proper name/hash expected by override_settings

def process_string_tag(tag):
    return tag

def process_int_tag(tag):
    try:
        return int(tag)
    except ValueError:
        errors.report(f"Warning: Invalid integer value: {tag}. Using default.", exc_info=False)
        return None

def process_float_tag(tag):
    try:
        return float(tag)
    except ValueError:
        errors.report(f"Warning: Invalid float value: {tag}. Using default.", exc_info=False)
        return None

def process_boolean_tag(tag):
    return tag.lower() == "true"

# Extended list of supported tags
prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag, # Can be used to override base output dir per job
    "outpath_grids": process_string_tag,   # Can be used to override base grid dir per job
    "filename_pattern": process_string_tag,# Custom pattern for individual images (uses placeholders)
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag, # Note: Style handling might need adjustment based on UI interaction
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag, # Less common, use sampler_name
    "sampler_name": process_string_tag, # Will be mapped later
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag,
    "eta": process_float_tag, # Added common parameter
    "sampler_s_churn": process_float_tag,
    "sampler_s_tmin": process_float_tag,
    "sampler_s_tmax": process_float_tag,
    "sampler_s_noise": process_float_tag,
    "override_settings_restore_afterwards": process_boolean_tag,
    # --- Placeholders for Extension Params (Require custom handling) ---
    # "lora": process_string_tag, # Example: --lora "mylora:0.8"
    # "controlnet_args": process_string_tag, # Example: --controlnet_args "[{'model': 'scribble', 'image': '...', ...}]"
}

# Regex for parameter sweeping: --param [start..end step s] or --param [v1, v2, v3]
sweep_pattern = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)\s*(?:step\s*(-?\d+(?:\.\d+)?))?\s*\]")
list_pattern = re.compile(r"\[\s*((?:[^,\[\]]+(?:\s*,\s*[^,\[\]]+)*)?)\s*\]") # Handles comma-separated values

def generate_sweep_values(sweep_match, tag_type):
    """Generates values from a sweep range."""
    start, end, step = sweep_match.groups()
    start = float(start)
    end = float(end)
    step = float(step) if step else 1.0

    values = []
    current = start
    # Handle potential floating point inaccuracies for comparison
    epsilon = abs(step) * 0.0001
    if step > 0:
        while current <= end + epsilon:
            values.append(current)
            current += step
    elif step < 0:
         while current >= end - epsilon:
            values.append(current)
            current += step
    else: # step is 0, just use start value
        values.append(start)

    # Convert back to int if the original tag expects int
    if tag_type == process_int_tag:
        return [int(round(v)) for v in values] # Round before int conversion
    return values

def generate_list_values(list_match, func):
    """Generates values from a comma-separated list."""
    content = list_match.group(1)
    if not content:
        return []
    vals_str = [s.strip() for s in content.split(',')]
    processed_vals = []
    for v_str in vals_str:
        try:
            processed = func(v_str)
            if processed is not None: # Check if processing func signaled an error
                 processed_vals.append(processed)
            else:
                 errors.report(f"Warning: Invalid value '{v_str}' in list, skipping.", exc_info=False)
        except Exception as e:
            errors.report(f"Warning: Error processing value '{v_str}' in list: {e}. Skipping.", exc_info=False)
    return processed_vals


def cmdargs(line):
    """Parses a line into arguments, handling comments, sweeps, and lists. Returns a LIST of job dictionaries."""
    # Strip comments first (allow inline comments starting with #)
    line = line.split('#', 1)[0].strip()
    if not line:
        return [] # Skip empty or comment-only lines

    try:
        args = shlex.split(line)
    except ValueError as e:
        # If shlex fails (e.g., unmatched quotes), treat the whole line as a prompt
        errors.report(f"Warning: Could not parse line with shlex: '{line}'. Treating as simple prompt. Error: {e}", exc_info=False)
        return [{"prompt": line}] # Return as a list containing one job

    pos = 0
    res = {}
    sweep_args = {} # Dictionary to store {tag: [value1, value2, ...]} for sweeping

    # Check if the first token looks like a command or a prompt
    is_command_line = args[0].startswith("--") if args else False

    if not is_command_line:
        # Treat the whole line as a prompt if it doesn't start with "--"
        res["prompt"] = line
        return [res] # Return as a list containing one job

    # --- Parsing Loop for command-line style ---
    while pos < len(args):
        arg = args[pos]

        # Handle prompt specially if it doesn't start with -- and we already parsed some args
        # Or if it explicitly uses --prompt or --negative_prompt
        is_prompt_tag = arg == "--prompt"
        is_neg_prompt_tag = arg == "--negative_prompt"
        is_generic_tag = arg.startswith("--") and not is_prompt_tag and not is_neg_prompt_tag

        if is_prompt_tag or is_neg_prompt_tag:
            tag = arg[2:]
            pos += 1
            if pos >= len(args):
                 errors.report(f"Error parsing line '{line}': Missing argument for command line option {arg}", exc_info=True)
                 raise ValueError(f"Missing argument for {arg}")

            prompt = args[pos]
            pos += 1
            # Consume subsequent tokens that don't start with "--" as part of the prompt
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt # Assign collected prompt
            continue # Continue parsing loop

        elif is_generic_tag:
            tag = arg[2:]
            func = prompt_tags.get(tag)
            if not func:
                errors.report(f"Error parsing line '{line}': Unknown commandline option: {arg}", exc_info=True)
                raise ValueError(f"Unknown commandline option: {arg}")

            pos += 1
            if pos >= len(args):
                 errors.report(f"Error parsing line '{line}': Missing argument for command line option {arg}", exc_info=True)
                 raise ValueError(f"Missing argument for {arg}")

            val_str = args[pos]
            pos += 1 # Move past value

            # Check for sweep/list syntax BEFORE processing the value
            sweep_match = sweep_pattern.match(val_str)
            list_match = list_pattern.match(val_str)

            if sweep_match:
                sweep_values = generate_sweep_values(sweep_match, func)
                if sweep_values:
                    sweep_args[tag] = sweep_values
                else:
                     errors.report(f"Warning: Invalid sweep range for {tag}: {val_str}. Skipping.", exc_info=False)

            elif list_match:
                list_values = generate_list_values(list_match, func)
                if list_values:
                    sweep_args[tag] = list_values # Treat list as a type of sweep
                else:
                    errors.report(f"Warning: Invalid list format or empty list for {tag}: {val_str}. Skipping.", exc_info=False)

            else: # Regular value processing
                try:
                    # Special handling for sampler name mapping
                    if tag == "sampler_name":
                        mapped_sampler = sd_samplers.samplers_map.get(val_str.lower())
                        if mapped_sampler:
                            res[tag] = mapped_sampler.name # Store the official name
                        else:
                            errors.report(f"Warning: Unknown sampler name: {val_str}. Using default.", exc_info=False)
                            res[tag] = None # Signal to use default
                    else:
                        processed_val = func(val_str)
                        if processed_val is not None: # Check if processor signaled error
                             res[tag] = processed_val

                except Exception as e:
                    errors.report(f"Error processing tag '{tag}' with value '{val_str}': {e}", exc_info=True)
                    raise ValueError(f"Error processing tag '{tag}': {e}")

        else: # Token doesn't start with --, assume it's part of a prompt at the start
             if "prompt" not in res:
                 # Collect the initial prompt part
                 prompt = arg
                 pos += 1
                 while pos < len(args) and not args[pos].startswith("--"):
                     prompt += " "
                     prompt += args[pos]
                     pos += 1
                 res["prompt"] = prompt
             else:
                 # This case should ideally not be reached if parsing logic is correct
                 errors.report(f"Warning: Unexpected token '{arg}' in line '{line}'. Ignoring.", exc_info=False)
                 pos += 1

    # --- Expand Sweeps ---
    if not sweep_args:
        # If no sweeps, return the single result dictionary in a list
        return [res] if res else []
    else:
        # Generate all combinations of sweep values
        expanded_jobs = []
        # Get sweep keys and their value lists
        sweep_keys = list(sweep_args.keys())
        sweep_value_lists = [sweep_args[key] for key in sweep_keys]

        # Generate Cartesian product of sweep values
        import itertools
        product_values = list(itertools.product(*sweep_value_lists))

        for value_combination in product_values:
            job_variation = res.copy() # Start with base args
            for i, key in enumerate(sweep_keys):
                job_variation[key] = value_combination[i] # Apply sweep value
            expanded_jobs.append(job_variation)

        # If the original line had no base args (only sweeps), ensure each job isn't empty
        if not res and expanded_jobs:
            return expanded_jobs
        elif not expanded_jobs and res: # Only base args, no valid sweeps generated
             return [res]
        elif not expanded_jobs and not res: # Empty line or only invalid sweeps
            return []
        else:
            return expanded_jobs


def load_prompt_file(file):
    if file is None:
        return None, gr.update()
    else:
        try:
            lines = [x.strip() for x in file.decode('utf-8', errors='ignore').split("\n")]
            return None, "\n".join(lines)
        except Exception as e:
            errors.report(f"Error loading prompt file: {e}", exc_info=True)
            return None, f"Error loading file: {e}"


class Script(scripts.Script):
    def title(self):
        return "Prompts from file or textbox v2 (Enhanced)"

    def ui(self, is_img2img):
        # --- UI Elements ---
        with gr.Accordion("Input Options", open=True):
            prompt_txt = gr.Textbox(label="List of prompt inputs / commands", lines=10, elem_id=self.elem_id("prompt_txt"))
            file = gr.File(label="Upload prompt inputs (file overrides textbox)", type='binary', elem_id=self.elem_id("file"))

        with gr.Accordion("Processing Options", open=True):
            checkbox_iterate = gr.Checkbox(label="Iterate seed every line/job", value=False, elem_id=self.elem_id("checkbox_iterate"))
            checkbox_iterate_batch = gr.Checkbox(label="Use same random seed for all lines (if main seed is -1)", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
            prompt_position = gr.Radio(["start", "end", "replace"], label="Combine main UI prompt with line prompt:", elem_id=self.elem_id("prompt_position"), value="start")
            neg_prompt_position = gr.Radio(["start", "end", "replace"], label="Combine main UI negative prompt with line negative prompt:", elem_id=self.elem_id("neg_prompt_position"), value="start")
            error_handling = gr.Dropdown(["Stop processing", "Log error and continue"], label="Action on error parsing/processing a line:", value="Stop processing", elem_id=self.elem_id("error_handling"))

        with gr.Accordion("Output Options", open=True):
            make_combined = gr.Checkbox(label="Make a combined image grid (if >1 image generated)", value=False, elem_id=self.elem_id("make_combined"))
            grid_filename_pattern = gr.Textbox(label="Combined grid filename pattern", placeholder="grid-{model_name}-{seed}.png | Placeholders: {prompt_slug}, {seed}, {model_name}, etc.", elem_id=self.elem_id("grid_filename_pattern"))
            save_job_list = gr.Checkbox(label="Save processed job list as JSON file", value=False, elem_id=self.elem_id("save_job_list"))

        # --- Event Handlers ---
        file.upload(fn=load_prompt_file, inputs=[file], outputs=[file, prompt_txt], show_progress=True)

        # Return list of UI components needed in run()
        return [checkbox_iterate, checkbox_iterate_batch, prompt_position, neg_prompt_position, error_handling, prompt_txt, make_combined, grid_filename_pattern, save_job_list]

    def run(self, p, checkbox_iterate, checkbox_iterate_batch, prompt_position, neg_prompt_position, error_handling, prompt_txt: str, make_combined, grid_filename_pattern, save_job_list):

        # --- 1. Preprocessing: Read Lines, Handle Variables and Comments ---
        lines = [x for x in (x.strip() for x in prompt_txt.splitlines())]

        variables = {}
        processed_lines = []
        variable_pattern = re.compile(r"^\s*\$(\w+)\s*=\s*(.*)$")

        # First pass: Extract variables and filter comments/empty lines
        for line in lines:
            if line.startswith('#') or line.startswith('//') or not line:
                continue # Skip comments and empty lines

            var_match = variable_pattern.match(line)
            if var_match:
                var_name = var_match.group(1)
                var_value = var_match.group(2).strip()
                variables[f"${var_name}"] = var_value
                print(f"Defined variable: ${var_name} = {var_value}")
            else:
                processed_lines.append(line) # Keep lines that are not variable definitions or comments

        # Second pass: Substitute variables
        final_lines = []
        for line in processed_lines:
            original_line = line
            for var_name, var_value in variables.items():
                 # Use regex to avoid replacing parts of words, ensure var_name is escaped
                 # Using simple replace for now, more robust regex might be needed
                 line = line.replace(var_name, var_value)
            final_lines.append(line)
            if line != original_line:
                 print(f"Substituted: {original_line} -> {line}")


        # --- 2. Job Expansion: Parse Lines, Handle Sweeps ---
        p.do_not_save_grid = True # We handle grid saving manually if requested

        all_jobs_expanded = []
        original_line_mapping = {} # Map expanded job index back to original line number

        print(f"Parsing {len(final_lines)} lines...")
        for i, line in enumerate(final_lines):
            if state.interrupted:
                break
            try:
                # cmdargs now returns a LIST of job dictionaries (due to sweeping)
                job_variations = cmdargs(line)
                if job_variations: # Check if cmdargs returned any valid jobs
                    for job_idx, job_args in enumerate(job_variations):
                         # Store mapping: index in all_jobs_expanded -> original line index i
                         original_line_mapping[len(all_jobs_expanded)] = i
                         all_jobs_expanded.append(job_args)
                elif line.strip(): # Report if a non-empty line resulted in no jobs (e.g., parse error)
                    print(f"Warning: Line {i+1} ('{line[:50]}...') did not generate any jobs. Skipping.")

            except Exception as e:
                errors.report(f"Failed to parse line {i+1}: '{line[:100]}...'. Error: {e}", exc_info=True)
                if error_handling == "Stop processing":
                    print("Stopping processing due to error.")
                    return Processed(p, [], p.seed, f"Error parsing line {i+1}")
                else:
                    print(f"Skipping line {i+1} due to error.")
                    continue # Skip to the next line

        if not all_jobs_expanded:
            print("No valid jobs found to process.")
            return Processed(p, [], p.seed, "No valid jobs processed.")

        # --- 3. Calculate Total Jobs & Initialize State ---
        initial_job_count = 0
        for job_args in all_jobs_expanded:
             initial_job_count += job_args.get("n_iter", p.n_iter) * job_args.get("batch_size", p.batch_size) # Rough estimate

        print(f"Will process {len(final_lines)} lines, expanded into {len(all_jobs_expanded)} jobs (approx. {initial_job_count} images).")

        # Initialize seed if needed
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))
            print(f"Set initial random seed to: {p.seed}")

        # Use a fixed seed for all jobs if iterate_batch is checked and seed was random
        initial_fixed_seed = p.seed if checkbox_iterate_batch else None

        state.job_count = len(all_jobs_expanded) # Total number of jobs (process_images calls)
        state.job_no = 0

        images = []
        all_prompts = []
        infotexts = []
        processed_jobs_data = [] # For saving job list

        # --- 4. Main Processing Loop ---
        for job_index, args in enumerate(all_jobs_expanded):
            if state.interrupted:
                break

            state.job = f"Job {state.job_no + 1}/{state.job_count} (Line {original_line_mapping[job_index]+1})"
            print(f"\n--- Processing Job {state.job_no + 1}/{state.job_count} (From Line {original_line_mapping[job_index]+1}) ---")
            print(f"Args: {args}")

            copy_p = copy.copy(p)

            # --- Apply Arguments to copy_p ---
            try:
                # Apply initial fixed seed if iterate_batch is selected
                if initial_fixed_seed is not None:
                    copy_p.seed = initial_fixed_seed

                # Apply arguments from the current job
                current_job_seed = args.get("seed", None) # Check for explicit seed in args

                # Store original paths before potentially overriding them
                original_outpath_samples = copy_p.outpath_samples
                original_outpath_grids = copy_p.outpath_grids
                original_filename_pattern = getattr(opts, 'samples_filename_pattern', '') # Get default pattern

                custom_outpath_samples = None
                custom_outpath_grids = None
                custom_filename_pattern = None

                # --- Prepare Overrides ---
                copy_p.override_settings = copy_p.override_settings.copy() # Ensure we don't modify the original p's overrides
                copy_p.override_settings['override_settings_restore_afterwards'] = True # Good practice

                for k, v in args.items():
                    if v is None: continue # Skip if processor function returned None (error)

                    # --- Special Handling ---
                    if k == "sd_model":
                        print(f"Switching model to: {v}")
                        copy_p.override_settings['sd_model_checkpoint'] = v
                    elif k == "outpath_samples":
                        custom_outpath_samples = v # Don't set directly, use override
                        print(f"Setting custom sample output path: {v}")
                        copy_p.override_settings['outpath_samples'] = v
                    elif k == "outpath_grids":
                         custom_outpath_grids = v
                         print(f"Setting custom grid output path: {v}")
                         copy_p.override_settings['outpath_grids'] = v
                    elif k == "filename_pattern":
                         custom_filename_pattern = v
                         print(f"Setting custom filename pattern: {v}")
                         copy_p.override_settings['samples_filename_pattern'] = v # Override the setting
                    elif k == "sampler_name":
                        # Find the sampler object by name
                        sampler_info = sd_samplers.find_sampler_by_name(v)
                        if sampler_info:
                            copy_p.sampler_name = sampler_info.name # Set the name
                            print(f"Setting sampler to: {sampler_info.name}")
                        else:
                            print(f"Warning: Sampler '{v}' not found. Using default.")
                    # --- Placeholder for Extension Args Handling ---
                    # elif k == "lora":
                    #     # Parse "name:weight" and apply using extension API
                    #     print(f"Applying LoRA: {v} (IMPLEMENTATION NEEDED)")
                    #     # Example: parse v, find lora, modify copy_p.prompt or use specific API
                    # elif k == "controlnet_args":
                    #     # Parse JSON string and apply using ControlNet API (likely via script_args)
                    #     print(f"Applying ControlNet args: {v} (IMPLEMENTATION NEEDED)")
                    #     # Example: parsed_args = json.loads(v); copy_p.script_args = parsed_args
                    # --- Standard Attributes ---
                    elif hasattr(copy_p, k):
                         print(f"Setting {k} = {v}")
                         setattr(copy_p, k, v)
                    elif k in copy_p.override_settings: # Check if it's an override setting
                         print(f"Overriding setting {k} = {v}")
                         copy_p.override_settings[k] = v
                    else:
                         print(f"Warning: Unknown argument '{k}'. Skipping.")


                # --- Handle Prompt Combination ---
                line_prompt = args.get("prompt", "")
                if line_prompt and p.prompt:
                    if prompt_position == "start":
                        copy_p.prompt = line_prompt + (" " if p.prompt else "") + p.prompt
                    elif prompt_position == "end":
                        copy_p.prompt = p.prompt + (" " if line_prompt else "") + line_prompt
                    elif prompt_position == "replace":
                         copy_p.prompt = line_prompt
                elif line_prompt:
                    copy_p.prompt = line_prompt
                # else: keep original p.prompt

                # --- Handle Negative Prompt Combination ---
                line_neg_prompt = args.get("negative_prompt", "")
                if line_neg_prompt and p.negative_prompt:
                     if neg_prompt_position == "start":
                         copy_p.negative_prompt = line_neg_prompt + (" " if p.negative_prompt else "") + p.negative_prompt
                     elif neg_prompt_position == "end":
                         copy_p.negative_prompt = p.negative_prompt + (" " if line_neg_prompt else "") + line_neg_prompt
                     elif neg_prompt_position == "replace":
                          copy_p.negative_prompt = line_neg_prompt
                elif line_neg_prompt:
                     copy_p.negative_prompt = line_neg_prompt
                # else: keep original p.negative_prompt


                # --- Seed Handling (Override/Iteration) ---
                if current_job_seed is not None:
                    copy_p.seed = int(current_job_seed) # Explicit seed overrides iteration
                    print(f"Using explicit seed from line: {copy_p.seed}")
                elif checkbox_iterate and initial_fixed_seed is None: # Only iterate if not using fixed batch seed
                    # Increment the *original* p's seed for the next iteration
                    # We use the *current* value of p.seed for this job
                    copy_p.seed = p.seed
                    print(f"Using iterated seed: {copy_p.seed}")
                    p.seed += args.get("batch_size", p.batch_size) * args.get("n_iter", p.n_iter)
                elif not checkbox_iterate and initial_fixed_seed is None:
                     # Use the main UI seed value (potentially random if -1 initially)
                     copy_p.seed = p.seed
                     print(f"Using main seed: {copy_p.seed}")
                # If initial_fixed_seed is set, copy_p.seed was already set earlier


                # --- Replace Placeholders in Custom Paths/Filenames ---
                job_info_for_placeholders = args.copy() # Get args for placeholder context
                job_info_for_placeholders['seed'] = copy_p.seed # Ensure seed is the one being used
                job_info_for_placeholders['sampler_name'] = copy_p.sampler_name
                job_info_for_placeholders['prompt'] = copy_p.prompt
                job_info_for_placeholders['negative_prompt'] = copy_p.negative_prompt
                job_info_for_placeholders['width'] = copy_p.width
                job_info_for_placeholders['height'] = copy_p.height
                # ... add other relevant params from copy_p if needed

                if custom_outpath_samples:
                    copy_p.override_settings['outpath_samples'] = replace_placeholders(custom_outpath_samples, job_info_for_placeholders, copy_p, original_line_mapping[job_index]+1)
                    print(f"Processed sample output path: {copy_p.override_settings['outpath_samples']}")
                if custom_outpath_grids:
                     copy_p.override_settings['outpath_grids'] = replace_placeholders(custom_outpath_grids, job_info_for_placeholders, copy_p, original_line_mapping[job_index]+1)
                     print(f"Processed grid output path: {copy_p.override_settings['outpath_grids']}")
                if custom_filename_pattern:
                     copy_p.override_settings['samples_filename_pattern'] = replace_placeholders(custom_filename_pattern, job_info_for_placeholders, copy_p, original_line_mapping[job_index]+1)
                     print(f"Processed filename pattern: {copy_p.override_settings['samples_filename_pattern']}")


                # --- Collect Job Data for Saving ---
                if save_job_list:
                    job_data = {
                        "line_number": original_line_mapping[job_index] + 1,
                        "original_args": args,
                        "processed_params": {
                             k: getattr(copy_p, k) for k in ["prompt", "negative_prompt", "seed", "subseed", "subseed_strength", "steps", "cfg_scale", "width", "height", "sampler_name", "batch_size", "n_iter"] if hasattr(copy_p, k)
                        },
                        "overrides": copy_p.override_settings.copy() # Save resolved overrides for this job
                    }
                    # Add other relevant params if needed
                    processed_jobs_data.append(job_data)


                # --- Execute Processing ---
                print(f"Starting image generation for job {state.job_no + 1}...")
                proc = process_images(copy_p) # This is where the images are generated

                # Check for banned prompts after generation
                if getattr(proc, 'comments', None) and BannedPrompt.flag in proc.comments:
                     print(f"Warning: Job {state.job_no + 1} triggered safety filter. Output might be blurred/black.")


                # --- Collect Results ---
                images += proc.images
                all_prompts += proc.all_prompts
                infotexts += proc.infotexts

                # --- Restore potentially overridden opts (though override_settings should handle this) ---
                # opts.outdir_samples = original_outpath_samples # Usually not needed if override_settings_restore_afterwards=True
                # opts.outdir_grids = original_outpath_grids
                # opts.samples_filename_pattern = original_filename_pattern


            except Exception as e:
                errors.report(f"Error processing job {state.job_no + 1} (Line {original_line_mapping[job_index]+1}): {e}", exc_info=True)
                if error_handling == "Stop processing":
                    print("Stopping processing due to error.")
                    # Try to return partial results if any
                    if images:
                         return Processed(p, images, p.seed, f"Stopped due to error on job {state.job_no + 1}", all_prompts=all_prompts, infotexts=infotexts)
                    else:
                         return Processed(p, [], p.seed, f"Stopped due to error on job {state.job_no + 1}")
                else:
                    print(f"Skipping job {state.job_no + 1} due to error.")
                    # Need to manually increment job_no if skipping AFTER the main loop increments it
                    # Handled by state.job_no increment below

            state.job_no += 1 # Increment job number AFTER processing attempt


        # --- 5. Post-Processing: Grid Generation & Job List Saving ---
        output_images = images

        # --- Generate Combined Grid ---
        if make_combined and len(images) > 1:
            print(f"Creating combined grid for {len(images)} images...")
            try:
                # Determine grid save directory
                is_img2img = getattr(p, "init_images", None) is not None
                grid_outpath_base = p.override_settings.get('outpath_grids', opts.outdir_grids or (opts.outdir_img2img_grids if is_img2img else opts.outdir_txt2img_grids))

                # Determine grid filename
                if grid_filename_pattern:
                     # Use the first job's info for placeholders in the grid filename
                     first_job_info = all_jobs_expanded[0].copy()
                     first_job_info['seed'] = infotexts[0].split("Seed: ")[1].split(",")[0] if infotexts else p.seed # Try to get actual seed
                     grid_filename = replace_placeholders(grid_filename_pattern, first_job_info, p, 0)
                else:
                     grid_filename = f"grid-{p.seed}" # Default fallback

                # Create the grid image
                grid_rows = math.ceil(len(images) / (opts.grid_max_images_per_grid if opts.grid_max_images_per_grid > 0 else p.batch_size * p.n_iter)) # Simple row calc
                combined_image = image_grid(images, batch_size=p.batch_size, rows=grid_rows) # Adjust batch_size?

                # Create combined infotext (simplified)
                full_infotext = f"Grid from {len(images)} images. Prompt file/textbox used.\n"
                full_infotext += f"First prompt: {all_prompts[0]}\n" if all_prompts else ""
                full_infotext += f"First infotext: {infotexts[0]}\n" if infotexts else ""
                # Consider adding parameter ranges or job count here

                # Save the grid
                save_image(
                    combined_image,
                    grid_outpath_base,
                    "", # Base name (subdir handled by outpath)
                    p.seed, # Use initial seed for grid filename?
                    prompt_txt[:1000], # Use input prompts as prompt? (Truncated)
                    opts.grid_format,
                    info=full_infotext,
                    grid=True,
                    filename_pattern=grid_filename, # Pass the generated filename pattern
                    p=p # Pass processing obj for potential extra info saving
                )
                print(f"Saved combined grid: {os.path.join(grid_outpath_base, grid_filename + '.' + opts.grid_format)}")

                # Add grid to the start of the results
                output_images.insert(0, combined_image)
                all_prompts.insert(0, "Combined Grid") # Placeholder prompt
                infotexts.insert(0, full_infotext)

            except Exception as e:
                errors.report("Error creating or saving combined grid image.", exc_info=True)


        # --- Save Processed Job List ---
        if save_job_list and processed_jobs_data:
             try:
                 # Determine save directory (use grid or samples dir?)
                 save_dir = p.override_settings.get('outpath_grids', opts.outdir_grids or opts.outdir_txt2img_grids) # Default to grid dir
                 os.makedirs(save_dir, exist_ok=True)
                 # Filename based on seed or timestamp
                 joblist_filename = os.path.join(save_dir, f"processed_jobs_{p.seed}.json")
                 with open(joblist_filename, 'w', encoding='utf-8') as f:
                     json.dump(processed_jobs_data, f, indent=4, ensure_ascii=False)
                 print(f"Saved processed job list to: {joblist_filename}")
                 # Add filename to infotexts?
                 if infotexts:
                     infotexts[0] += f"\nProcessed job list saved to: {joblist_filename}"
             except Exception as e:
                 errors.report("Error saving processed job list.", exc_info=True)


        # --- Return Results ---
        print("Processing finished.")
        # Use the first infotext for the main output, others are attached
        main_info = infotexts[0] if infotexts else ""

        return Processed(p, output_images, p.seed, main_info, all_prompts=all_prompts, infotexts=infotexts)
