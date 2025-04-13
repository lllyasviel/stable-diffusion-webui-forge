
import datetime
import glob
import html
import os
import inspect
from contextlib import closing
from collections import deque, defaultdict # Added defaultdict for commented code in train_hypernetwork
from statistics import stdev, mean
from typing import List, Dict, Optional, Tuple, Any, Callable, Union # Added Union

import torch
import tqdm # type: ignore # Assuming tqdm is available
from einops import rearrange, repeat # type: ignore # Assuming einops is available
from torch import einsum
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_
from torch.optim.optimizer import Optimizer # For type hinting
from torch.cuda.amp import GradScaler # For type hinting

# --- Assumed External Imports (Placeholders & Error Handling) ---
# These are placeholders based on the original code. They will likely fail
# outside the original execution environment (e.g., Stable Diffusion Web UI).
try:
    from backend.nn.unet import default # type: ignore
    from modules import devices, sd_models, shared, sd_samplers, hashes, sd_hijack_checkpoint, errors # type: ignore
    from modules.textual_inversion import textual_inversion # type: ignore
    from modules import processing # type: ignore # Needed for train_hypernetwork
    from modules import images # type: ignore # Needed for train_hypernetwork
    from modules.textual_inversion import dataset # type: ignore # Needed for train_hypernetwork
    from modules.textual_inversion import saving_settings # type: ignore # Needed for train_hypernetwork
    # LearnRateScheduler is used but not imported in original. Assume it's in textual_inversion?
    try:
        from modules.textual_inversion.learn_schedule import LearnRateScheduler # type: ignore
    except ImportError:
        # Dummy placeholder if LearnRateScheduler is not found
        print("Warning: LearnRateScheduler not found. Using dummy placeholder.")
        class LearnRateScheduler:
            def __init__(self, learn_rate, steps, initial_step, verbose=True):
                self.learn_rate = learn_rate
                self.steps = steps
                self.initial_step = initial_step
                self.finished = False
            def apply(self, optimizer, step): pass
            def step(self, step): pass

except ImportError as e:
    print(f"Warning: Failed to import one or more external modules: {e}. Using dummy placeholders.")
    print("         The code, especially training functions, will NOT run correctly.")

    # Define dummy types or use Any if imports fail in a standalone context
    Any = type(Any) # type: ignore # Redefine Any locally
    default = lambda x, y: x if x is not None else y # type: ignore

    class DummyOpts: # More detailed dummy opts based on usage
        hypernetwork_dir = "./models/hypernetworks" # Example path
        pin_memory = False
        training_image_repeats_per_epoch = 1
        save_training_settings_to_txt = False
        training_enable_tensorboard = False
        training_tensorboard_save_images = False
        samples_format = "png"
        unload_models_when_training = False
        print_hypernet_extra = False
        save_optimizer_state = False

    class DummyShared:
        opts = DummyOpts()
        hypernetworks: Dict[str, str] = {}
        loaded_hypernetworks: List['Hypernetwork'] = []
        state = type('obj', (object,), {'job': '', 'textinfo': '', 'job_count': 0, 'job_no': 0, 'interrupted': False, 'assign_current_image': lambda img: None})()
        sd_model = type('obj', (object,), {
            'cond_stage_model': type('obj', (object,), {'to': lambda dev: None, '__call__': lambda txt: torch.randn(1, 77, 768)})(),
            'first_stage_model': type('obj', (object,), {'to': lambda dev: None})(),
            'weighted_forward': lambda x, c, w: (torch.tensor(0.5),),
            'forward': lambda x, c: (torch.tensor(0.5),)
        })()
        cmd_opts = type('obj', (object,), {'hypernetwork_dir': './models/hypernetworks'})() # Add cmd_opts used in create_hypernetwork
        def reload_hypernetworks(self): print("Dummy reload_hypernetworks called")


    shared = DummyShared() # type: ignore

    class DummyDevices:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def torch_npu_set_device(self): pass
        def cond_cast_unet(self, x): return x
        def cond_cast_float(self, x): return x.float()
        def autocast(self): return torch.autocast(self.device.type) # Use 'cuda' if available

    devices = DummyDevices() # type: ignore

    class DummyHashes:
        def sha256(self, *args): return "dummyhash123"
    hashes = DummyHashes() # type: ignore

    class DummyErrors:
        def report(self, *args, **kwargs): print(f"Error Reported: {args}, {kwargs}")
    errors = DummyErrors() # type: ignore

    class DummySDHijackCheckpoint:
        def add(self): pass
        def remove(self): pass
    sd_hijack_checkpoint = DummySDHijackCheckpoint() # type: ignore

    class DummySDModels:
        def select_checkpoint(self):
             return type('obj', (object,), {'model_name': 'dummy_model.ckpt', 'shorthash': 'dummyhash456'})()
    sd_models = DummySDModels() # type: ignore

    class DummyTextualInversion:
        class DummyTemplates:
            def get(self, name, default=None):
                 return type('obj', (object,), {'path': 'dummy_template.txt'})()
        textual_inversion_templates = DummyTemplates()
        def validate_train_inputs(*args, **kwargs): print("Dummy validate_train_inputs called")
        def tensorboard_setup(log_dir): print(f"Dummy tensorboard_setup called for {log_dir}"); return None
        def tensorboard_add(*args, **kwargs): pass
        def write_loss(*args, **kwargs): pass
        def tensorboard_add_image(*args, **kwargs): pass
        class DummyDataset: # Placeholder for dataset module
            class PersonalizedBase: # Placeholder class
                def __init__(self, *args, **kwargs):
                    self.batch_size = kwargs.get('batch_size', 1)
                    self.gradient_step = kwargs.get('gradient_step', 1)
                    self.latent_sampling_method = kwargs.get('latent_sampling_method', 'once')
                    self.indexes = list(range(10)) # Dummy data length
                def __len__(self): return len(self.indexes)

            class PersonalizedDataLoader: # Placeholder class
                 def __init__(self, ds, *args, **kwargs): self.ds = ds; self.batch_size = ds.batch_size
                 def __iter__(self): # Dummy iterator
                     batch_data = type('obj', (object,), {
                         'latent_sample': torch.randn(self.batch_size, 4, 64, 64),
                         'weight': torch.ones(self.batch_size),
                         'cond_text': [f'prompt {i}' for i in range(self.batch_size)],
                         'cond': [torch.randn(77, 768) for _ in range(self.batch_size)]
                     })()
                     for _ in range(len(self.ds) // self.batch_size): yield batch_data

        dataset = DummyDataset()

    textual_inversion = DummyTextualInversion() # type: ignore

    class DummyProcessing: # Placeholder for processing module
        class StableDiffusionProcessingTxt2Img: # Placeholder class
             def __init__(self, *args, **kwargs):
                 self.prompt = ""
                 self.negative_prompt = ""
                 self.steps = 20
                 self.sampler_name = "Euler"
                 self.cfg_scale = 7.0
                 self.seed = -1
                 self.width = 512
                 self.height = 512
                 self.disable_extra_networks = False
             def __enter__(self): return self
             def __exit__(self, exc_type, exc_val, exc_tb): pass
        def process_images(p): # Placeholder function
             print("Dummy process_images called")
             img = torch.rand(3, p.height, p.width) # Dummy image tensor
             infotext = f"Prompt: {p.prompt}, Steps: {p.steps}"
             # Convert tensor to PIL Image for saving simulation
             from PIL import Image
             img_pil = Image.fromarray((img.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy())
             return type('obj', (object,), {'images': [img_pil], 'infotexts': [infotext]})()

    processing = DummyProcessing() # type: ignore

    class DummyImages: # Placeholder for images module
        def save_image(*args, **kwargs):
             print(f"Dummy save_image called. Forced filename: {kwargs.get('forced_filename')}")
             return f"path/to/{kwargs.get('forced_filename', 'image')}.png", "dummy text info"
    images = DummyImages() # type: ignore

    class DummySamplers: # Placeholder for sd_samplers
         samplers_map = {"euler": "Euler", "ddim": "DDIM"} # Example
    sd_samplers = DummySamplers() # type: ignore

    class DummySavingSettings: # Placeholder for saving_settings
         def save_settings_to_file(*args, **kwargs): print("Dummy save_settings_to_file called")
    saving_settings = DummySavingSettings() # type: ignore

    # Placeholder LearnRateScheduler if not imported above
    if 'LearnRateScheduler' not in globals():
        print("Warning: LearnRateScheduler not found. Using dummy placeholder.")
        class LearnRateScheduler:
            def __init__(self, learn_rate, steps, initial_step, verbose=True):
                self.learn_rate = learn_rate
                self.steps = steps
                self.initial_step = initial_step
                self.finished = False
            def apply(self, optimizer, step): pass
            def step(self, step): pass


# --- Constants ---
DEFAULT_DROPOUT_RATE = 0.3

# --- Global Dictionaries ---
optimizer_dict: Dict[str, Callable[..., Optimizer]] = {
    optim_name: cls_obj for optim_name, cls_obj in inspect.getmembers(torch.optim, inspect.isclass)
    if issubclass(cls_obj, torch.optim.Optimizer) and optim_name != "Optimizer"
}

# --- Core Classes ---

class HypernetworkModule(torch.nn.Module):
    """
    A single module within the Hypernetwork, representing one layer transformation
    for Key or Value projection in the attention mechanism. Includes logic for
    layer structure, activation, normalization, dropout, and weight initialization.
    """
    activation_dict: Dict[str, Callable[[], torch.nn.Module]] = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    # Dynamically add activations from torch.nn.modules.activation
    activation_dict.update({
        cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation)
        if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'
    })

    # Weight initialization functions map
    weight_initializers: Dict[str, Callable[[torch.Tensor, torch.Tensor, Optional[str]], None]] = {
        'Normal': lambda w, b, af: (normal_(w, mean=0.0, std=0.01), normal_(b, mean=0.0, std=0)),
        'XavierUniform': lambda w, b, af: (xavier_uniform_(w), zeros_(b)),
        'XavierNormal': lambda w, b, af: (xavier_normal_(w), zeros_(b)),
        'KaimingUniform': lambda w, b, af: (kaiming_uniform_(w, nonlinearity='leaky_relu' if af == 'leakyrelu' else 'relu'), zeros_(b)),
        'KaimingNormal': lambda w, b, af: (kaiming_normal_(w, nonlinearity='leaky_relu' if af == 'leakyrelu' else 'relu'), zeros_(b)),
    }

    def __init__(self,
                 dim: int,
                 state_dict: Optional[Dict[str, torch.Tensor]] = None,
                 layer_structure: Optional[List[Union[int, float]]] = None, # Allow float for multipliers
                 activation_func: Optional[str] = None,
                 weight_init: str = 'Normal',
                 add_layer_norm: bool = False,
                 activate_output: bool = False,
                 dropout_structure: Optional[List[float]] = None):
        """
        Initializes the HypernetworkModule.

        Args:
            dim: The base dimension size for the layers.
            state_dict: An optional state dictionary to load weights.
            layer_structure: A list defining the multiplier for layer dimensions (e.g., [1, 2, 1]). Defaults to [1, 2, 1].
            activation_func: The name of the activation function to use between layers. Defaults to 'linear'.
            weight_init: The name of the weight initialization method. Defaults to 'Normal'.
            add_layer_norm: Whether to add Layer Normalization after linear layers (and before activation).
            activate_output: Whether to apply the activation function to the final layer's output.
            dropout_structure: A list defining dropout probabilities after each layer element. Length must match layer_structure. Defaults to all zeros.
        """
        super().__init__()

        self.multiplier = 1.0
        self.layer_structure = layer_structure or [1.0, 2.0, 1.0]
        self.activation_func_name = activation_func or "linear" # Store name for init

        assert self.layer_structure[0] == 1, "Layer structure must start with 1!"
        assert self.layer_structure[-1] == 1, "Layer structure must end with 1!"

        if dropout_structure is None:
            self.dropout_structure = [0.0] * len(self.layer_structure)
        else:
             self.dropout_structure = dropout_structure
        assert len(self.dropout_structure) == len(self.layer_structure), \
             f"Dropout structure length ({len(self.dropout_structure)}) must match layer structure ({len(self.layer_structure)})"

        linears = []
        for i in range(len(self.layer_structure) - 1):
            input_dim = int(dim * self.layer_structure[i])
            output_dim = int(dim * self.layer_structure[i+1])

            # Add Linear layer
            linear_layer = torch.nn.Linear(input_dim, output_dim)
            linears.append(linear_layer)

            # Add LayerNorm (if enabled) before activation
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(output_dim)) # Norm applied on the output dim

            # Add Activation Function (conditionally)
            is_last_layer_transform = (i >= len(self.layer_structure) - 2) # Is this the linear layer before the final output?
            if self.activation_func_name != "linear" and (not is_last_layer_transform or activate_output):
                if self.activation_func_name in self.activation_dict:
                    linears.append(self.activation_dict[self.activation_func_name]())
                else:
                    raise ValueError(f"Unsupported activation function: {self.activation_func_name}")

            # Add Dropout (if enabled for the layer *output*)
            # dropout_structure[i+1] corresponds to dropout *after* the i-th transformation block (Linear, Norm, Act)
            dropout_prob = self.dropout_structure[i+1]
            if dropout_prob > 0:
                assert 0 < dropout_prob < 1, f"Dropout probability must be between 0 and 1 (exclusive), got {dropout_prob} for layer {i+1}"
                linears.append(torch.nn.Dropout(p=dropout_prob))

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            # Potential issue: state_dict keys depend heavily on the exact sequence of layers added.
            # fix_old_state_dict might need adjustment based on how Norm/Act/Dropout affect indices.
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            self.initialize_weights(weight_init)

        # Move to device - ideally done by the parent Hypernetwork class after creation
        try:
            # devices.torch_npu_set_device() # NPU specific call, might not be needed/available
            pass
        except AttributeError:
             pass # Silently ignore if function doesn't exist
        # self.to(devices.device) # Let parent class handle device placement

    def initialize_weights(self, weight_init: str):
        """Initializes weights of Linear and LayerNorm layers based on the chosen method."""
        if weight_init not in self.weight_initializers and weight_init != 'Normal': # 'Normal' is default/fallback
             print(f"Warning: Unsupported weight initialization method '{weight_init}'. Using 'Normal'.")
             weight_init = 'Normal' # Fallback to normal if unknown

        init_func = self.weight_initializers.get(weight_init)

        for layer in self.linear:
            if isinstance(layer, torch.nn.Linear):
                if init_func:
                    # Pass activation func name for Kaiming
                    init_func(layer.weight.data, layer.bias.data, self.activation_func_name)
                else: # Fallback (should not happen with the check above, but belts and suspenders)
                    normal_(layer.weight.data, mean=0.0, std=0.01)
                    normal_(layer.bias.data, mean=0.0, std=0)
            elif isinstance(layer, torch.nn.LayerNorm):
                 # Standard LayerNorm init (often 1 for weight, 0 for bias)
                 normal_(layer.weight.data, mean=1.0, std=0.01) # Sometimes init gain to 1
                 normal_(layer.bias.data, mean=0.0, std=0)

    def fix_old_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Corrects keys for compatibility with older saved formats.
        NOTE: This is brittle and depends heavily on the old vs new layer structure.
        It assumes old keys were 'linear1', 'linear2' and maps them to nn.Sequential indices.
        """
        # --- This mapping needs careful verification based on actual layer indices ---
        # Example: If layers are [Linear(0), LayerNorm(1), ReLU(2), Dropout(3), Linear(4), ...]
        changes = {
            'linear1.bias': 'linear.0.bias',     # Assuming first Linear is at index 0
            'linear1.weight': 'linear.0.weight',
            # Adjust index '2' or '4' based on whether Norm/Act/Dropout exist after linear.0
            'linear2.bias': 'linear.4.bias',     # Example: If Linear, Norm, Act, Dropout, Linear -> index 4
            'linear2.weight': 'linear.4.weight',
            # Add more mappings if needed for deeper networks or different old formats
        }
        # --- End mapping section ---

        keys_to_delete = []
        new_items = {}
        for fr, to in changes.items():
            if fr in state_dict:
                # Check if target key already exists (might happen with overlapping/incorrect maps)
                if to in state_dict:
                     print(f"Warning: Target key '{to}' already exists while migrating '{fr}'. Skipping migration for this key.")
                     continue
                print(f"Migrating old key '{fr}' to '{to}'")
                new_items[to] = state_dict[fr]
                keys_to_delete.append(fr)

        for key in keys_to_delete:
            del state_dict[key]
        state_dict.update(new_items)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the hypernetwork module transformation with a residual connection."""
        # Apply multiplier during inference/eval only
        mult = self.multiplier if not self.training else 1.0
        residual = self.linear(x)
        return x + residual * mult

    def trainables(self) -> List[torch.nn.Parameter]:
        """Returns a list of trainable parameters (weights and biases) in this module."""
        # Use self.parameters() for simplicity unless the specific structure is required downstream
        # return list(self.parameters())
        params = []
        for layer in self.linear:
            if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
                # LayerNorm parameters should also be trained
                params.extend(list(layer.parameters()))
        return params


def parse_dropout_structure(layer_structure: List[Union[int, float]],
                           use_dropout: bool,
                           last_layer_dropout: bool) -> List[float]:
    """
    Generates a dropout probability structure based on the layer structure multipliers.
    The list returned corresponds to dropout probability *after* each transformation stage.

    Args:
        layer_structure: The list of multipliers defining network layers (e.g., [1, 2, 1]).
        use_dropout: Whether dropout should be enabled at all.
        last_layer_dropout: Whether dropout should be applied after the second-to-last linear layer transformation block.

    Returns:
        A list of dropout probabilities, aligned with layer_structure elements (len matches).
        Index `i+1` corresponds to dropout after the transformation involving `layer_structure[i]` -> `layer_structure[i+1]`.
    """
    if not layer_structure: return [] # Handle empty case
    n_layers = len(layer_structure)
    if not use_dropout:
        return [0.0] * n_layers

    # dropout_values[i] is the dropout prob *after* the block ending at layer_structure[i]
    # Usually no dropout after input (index 0) or final output (index n_layers-1)
    dropout_values = [0.0] * n_layers

    # Apply dropout after intermediate transformation blocks
    # Loop from the first hidden layer output up to the second-to-last output
    for i in range(1, n_layers - 1): # Indices 1 to n_layers-2
         # Check if this is the second-to-last layer transformation block
         is_second_last_block = (i == n_layers - 2)
         if not is_second_last_block:
             # Apply default dropout to intermediate blocks
             dropout_values[i] = DEFAULT_DROPOUT_RATE
         else:
             # Apply dropout based on the last_layer_dropout flag
             dropout_values[i] = DEFAULT_DROPOUT_RATE if last_layer_dropout else 0.0

    # Ensure dropout[0] (after input) and dropout[-1] (after output) are 0
    dropout_values[0] = 0.0
    dropout_values[-1] = 0.0

    return dropout_values


class Hypernetwork:
    """
    Represents a Hypernetwork, containing modules for different attention layer dimensions
    and handling saving, loading, and training state.
    """
    filename: Optional[str]
    name: Optional[str]
    layers: Dict[int, Tuple[HypernetworkModule, HypernetworkModule]] # dim -> (K_module, V_module)
    step: int
    sd_checkpoint: Optional[str]
    sd_checkpoint_name: Optional[str]
    layer_structure: List[Union[int, float]]
    activation_func: Optional[str]
    weight_init: str
    add_layer_norm: bool
    use_dropout: bool
    activate_output: bool
    last_layer_dropout: bool # Specific dropout config flag used if structure not provided
    dropout_structure: List[float]
    optimizer_name: Optional[str]
    optimizer_state_dict: Optional[Dict[str, Any]]
    optional_info: Optional[Any] # For storing miscellaneous metadata like description

    def __init__(self,
                 name: Optional[str] = None,
                 enable_sizes: Optional[List[int]] = None,
                 layer_structure: Optional[List[Union[int, float]]] = None,
                 activation_func: Optional[str] = None,
                 weight_init: Optional[str] = None, # Changed Optional, default below
                 add_layer_norm: bool = False,
                 use_dropout: bool = False,
                 activate_output: bool = False,
                 last_layer_dropout: bool = True, # Default from original kwargs.get
                 dropout_structure: Optional[List[float]] = None,
                 **kwargs): # Catch any other potential legacy args like 'optional_info'
        """
        Initializes the Hypernetwork.

        Args:
            name: Name of the hypernetwork.
            enable_sizes: List of attention dimension sizes to create modules for (e.g., [320, 640, 1280]).
            layer_structure: Structure definition (list of multipliers) for the modules. Defaults to [1, 2, 1].
            activation_func: Activation function name for the modules. Defaults to None (interpreted as 'linear').
            weight_init: Weight initialization method name for the modules. Defaults to 'Normal'.
            add_layer_norm: Whether modules should use Layer Normalization.
            use_dropout: Whether modules should use Dropout (used to generate structure if not provided).
            activate_output: Whether modules should activate their final output layer.
            last_layer_dropout: Controls dropout specifically before the last layer transformation (used if structure not provided).
            dropout_structure: Explicit dropout structure; overrides calculation if provided. Length must match layer_structure.
            **kwargs: Allows capturing extra fields like 'optional_info'.
        """
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure or [1.0, 2.0, 1.0]
        self.activation_func = activation_func # Can be None -> linear
        self.weight_init = weight_init or 'Normal'
        self.add_layer_norm = add_layer_norm
        self.activate_output = activate_output
        self.optional_info = kwargs.get('optional_info', None) # Store extra info if present

        # Determine dropout configuration carefully
        self.last_layer_dropout = last_layer_dropout # Store the flag regardless
        if dropout_structure is not None:
            if len(dropout_structure) != len(self.layer_structure):
                 raise ValueError(f"Provided dropout_structure length ({len(dropout_structure)}) must match layer_structure length ({len(self.layer_structure)})")
            self.dropout_structure = dropout_structure
            # Infer use_dropout if structure is provided and contains non-zero values
            self.use_dropout = any(p > 0 for p in self.dropout_structure)
        else:
            # Generate structure based on flags if not explicitly provided
            self.use_dropout = use_dropout # Store the flag
            self.dropout_structure = parse_dropout_structure(
                self.layer_structure, self.use_dropout, self.last_layer_dropout
            )

        self.optimizer_name = None # Will be loaded or set during training
        self.optimizer_state_dict = None # Will be loaded or set during training

        # Create K and V modules for each specified dimension size
        enable_sizes = enable_sizes or []
        for size in enable_sizes:
            if not isinstance(size, int) or size <= 0:
                 print(f"Warning: Skipping invalid size in enable_sizes: {size}")
                 continue
            # Create modules but don't move to device yet
            module_k = HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure)
            module_v = HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure)
            self.layers[size] = (module_k, module_v)

        # Move the whole network to the correct device AFTER modules are created
        self.to(devices.device)
        self.eval() # Set to evaluation mode by default


    def weights(self) -> List[torch.nn.Parameter]:
        """Returns a list of all unique trainable parameters in the hypernetwork."""
        # Use a set to avoid duplicates if parameters are shared (though unlikely here)
        # return list({p for layers_tuple in self.layers.values() for layer_module in layers_tuple for p in layer_module.parameters()})
        # Simpler version assuming no shared params:
        all_params = []
        for mod_k, mod_v in self.layers.values():
             all_params.extend(mod_k.parameters())
             all_params.extend(mod_v.parameters())
        # Remove duplicates manually if needed, e.g., using dict.fromkeys
        return list(dict.fromkeys(all_params).keys())


    def train(self, mode: bool = True):
        """Sets the hypernetwork modules to training or evaluation mode."""
        # This method doesn't need to return self based on original code
        for layers_tuple in self.layers.values():
            for layer_module in layers_tuple:
                layer_module.train(mode=mode)
                # requires_grad is managed by torch based on train/eval mode,
                # but setting it explicitly ensures consistency if desired.
                for param in layer_module.parameters():
                    param.requires_grad_(mode) # Use requires_grad_ for in-place modification

    def to(self, device: torch.device):
        """Moves all hypernetwork modules to the specified device."""
        for layers_tuple in self.layers.values():
            for layer_module in layers_tuple:
                layer_module.to(device)
        return self # Allow chaining

    def set_multiplier(self, multiplier: float):
        """Sets the output multiplier for all modules."""
        for layers_tuple in self.layers.values():
            for layer_module in layers_tuple:
                layer_module.multiplier = multiplier
        return self # Allow chaining

    def eval(self):
        """Sets the hypernetwork modules to evaluation mode (disables dropout, sets requires_grad=False)."""
        self.train(mode=False)
        # Original code also set requires_grad=False explicitly in eval
        for param in self.weights(): # Get unique params
            param.requires_grad_(False)


    def save(self, filename: str):
        """Saves the hypernetwork state and optionally the optimizer state."""
        state_dict = {}

        # Save module state dicts
        for k, v_tuple in self.layers.items():
            if isinstance(k, int): # Ensure only integer keys are saved
                 state_dict[k] = (v_tuple[0].state_dict(), v_tuple[1].state_dict())
            else:
                 print(f"Warning: Skipping non-integer key '{k}' during hypernetwork save.")


        # Save metadata
        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['layer_structure'] = self.layer_structure
        state_dict['activation_func'] = self.activation_func
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['weight_initialization'] = self.weight_init
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name
        state_dict['activate_output'] = self.activate_output
        state_dict['use_dropout'] = self.use_dropout # Save the flag used for potential generation
        state_dict['dropout_structure'] = self.dropout_structure # Save the actual structure used
        # Infer last_layer_dropout from the saved structure for potential backward compat?
        state_dict['last_layer_dropout'] = (self.dropout_structure[-2] > 0) if len(self.dropout_structure) >= 2 else False
        state_dict['optional_info'] = self.optional_info

        # Save main hypernetwork file
        try:
            torch.save(state_dict, filename)
            print(f"Hypernetwork saved to: {filename}")
        except Exception as e:
            print(f"Error saving hypernetwork state to {filename}: {e}")
            errors.report(f"Error saving hypernetwork {self.name}", exc_info=True)
            return # Don't proceed to save optimizer if main save failed


        # Save optimizer state separately if configured and available
        # Check shared.opts exists and has the attribute
        save_optim = getattr(shared.opts, 'save_optimizer_state', False)
        if save_optim and self.optimizer_state_dict and self.optimizer_name:
            optimizer_saved_dict = {
                'optimizer_name': self.optimizer_name,
                'optimizer_state_dict': self.optimizer_state_dict,
                'hash': self.shorthash() # Include hash for verification on load
            }
            optim_filename = filename + '.optim'
            try:
                torch.save(optimizer_saved_dict, optim_filename)
                print(f"Optimizer state saved to: {optim_filename}")
            except Exception as e:
                print(f"Error saving optimizer state to {optim_filename}: {e}")
                errors.report(f"Error saving optimizer state for {self.name}", exc_info=True)


    def load(self, filename: str):
        """Loads the hypernetwork state and optionally the optimizer state from files."""
        if not os.path.exists(filename):
             raise FileNotFoundError(f"Hypernetwork file not found: {filename}")

        self.filename = filename
        # Set name from filename if not already set (e.g., during init)
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        # Load metadata (with defaults for backward compatibility)
        self.layer_structure = state_dict.get('layer_structure', [1.0, 2.0, 1.0])
        self.activation_func = state_dict.get('activation_func', None) # None is valid (linear)
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        # activate_output default was True in original code, maintain that?
        self.activate_output = state_dict.get('activate_output', True)
        self.optional_info = state_dict.get('optional_info', None)


        # Load dropout info carefully, prioritizing explicit structure
        loaded_dropout_structure = state_dict.get('dropout_structure', None)
        loaded_use_dropout = state_dict.get('use_dropout', False) # Fallback flag
        loaded_last_layer_dropout = state_dict.get('last_layer_dropout', False) # Fallback flag

        if loaded_dropout_structure is not None:
             # Verify length compatibility if possible
             if len(loaded_dropout_structure) == len(self.layer_structure):
                  self.dropout_structure = loaded_dropout_structure
             else:
                  print(f"Warning: Loaded dropout_structure length ({len(loaded_dropout_structure)}) "
                        f"mismatches layer_structure length ({len(self.layer_structure)}). "
                        f"Regenerating dropout structure.")
                  # Need to decide which flags to use for regeneration
                  self.use_dropout = any(p > 0 for p in loaded_dropout_structure) # Infer use_dropout
                  self.last_layer_dropout = (loaded_dropout_structure[-2] > 0) if len(loaded_dropout_structure) >= 2 else False # Infer flag
                  self.dropout_structure = parse_dropout_structure(
                       self.layer_structure, self.use_dropout, self.last_layer_dropout
                  )
             self.use_dropout = any(p > 0 for p in self.dropout_structure) # Set definitive flag
        else:
             # Fallback to old flags if structure not present
             print("Warning: 'dropout_structure' not found in state_dict. Generating from flags.")
             self.use_dropout = loaded_use_dropout
             self.last_layer_dropout = loaded_last_layer_dropout
             self.dropout_structure = parse_dropout_structure(
                 self.layer_structure, self.use_dropout, self.last_layer_dropout
             )

        # Print loaded info if desired (check shared.opts exists)
        print_extra = getattr(shared.opts, 'print_hypernet_extra', False)
        if print_extra:
            print(f"--- Loading hypernetwork: {self.name} ---")
            if self.optional_info is not None: print(f"  INFO: {self.optional_info}")
            print(f"  Layer structure: {self.layer_structure}")
            print(f"  Activation function: {self.activation_func}")
            print(f"  Weight initialization: {self.weight_init}")
            print(f"  Layer norm: {self.add_layer_norm}")
            print(f"  Activate last layer: {self.activate_output}")
            print(f"  Use dropout flag: {self.use_dropout}")
            print(f"  Last layer dropout flag (from load): {loaded_last_layer_dropout}") # Show potentially inaccurate flag
            print(f"  Dropout structure (final): {self.dropout_structure}")
            print(f"  Steps: {state_dict.get('step', 0)}")
            print(f"  SD Checkpoint: {state_dict.get('sd_checkpoint_name', 'N/A')} ({state_dict.get('sd_checkpoint', 'N/A')})")


        # Load optimizer state if available and hash matches
        optim_filename = self.filename + '.optim'
        optimizer_saved_dict = {}
        optim_load_success = False
        if os.path.exists(optim_filename):
            try:
                optimizer_saved_dict = torch.load(optim_filename, map_location='cpu')
                optim_load_success = True # Assume success unless hash fails
            except Exception as e:
                 print(f"Warning: Could not load optimizer state from {optim_filename}: {e}")
                 errors.report(f"Error loading optimizer state for {self.name}", exc_info=True)


        if optim_load_success and self.shorthash() == optimizer_saved_dict.get('hash', None):
            self.optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
            # Default to AdamW if name missing but state dict loaded (unlikely but possible)
            self.optimizer_name = optimizer_saved_dict.get('optimizer_name', 'AdamW')
            if self.optimizer_state_dict and print_extra:
                print(f"  Loaded existing optimizer state ({self.optimizer_name}).")
        else:
            self.optimizer_state_dict = None
            # Default to AdamW if no optim state loaded or hash mismatch
            self.optimizer_name = "AdamW"
            if print_extra:
                if optim_load_success and optimizer_saved_dict: # If file loaded but hash failed
                     print("  Optimizer checkpoint hash mismatch or missing hash. Ignoring saved optimizer state.")
                elif os.path.exists(optim_filename): # If file exists but load failed
                     print("  Could not load optimizer checkpoint file. Initializing new optimizer state.")
                else: # If file doesn't exist
                     print("  No saved optimizer state found. Initializing new optimizer state.")


        # Recreate modules using loaded state dicts and metadata
        self.layers = {} # Clear existing layers before loading
        for size, sd_tuple in state_dict.items():
            if isinstance(size, int):
                if not isinstance(sd_tuple, (tuple, list)) or len(sd_tuple) != 2:
                    print(f"Warning: Skipping invalid state dict entry for size {size}. Expected tuple of two state_dicts.")
                    continue
                sd_k, sd_v = sd_tuple
                try:
                    # Pass loaded metadata to the module constructor
                    module_k = HypernetworkModule(
                        size, sd_k, self.layer_structure, self.activation_func, self.weight_init,
                        self.add_layer_norm, self.activate_output, self.dropout_structure
                    )
                    module_v = HypernetworkModule(
                        size, sd_v, self.layer_structure, self.activation_func, self.weight_init,
                        self.add_layer_norm, self.activate_output, self.dropout_structure
                    )
                    self.layers[size] = (module_k, module_v)
                except Exception as e:
                     print(f"Error creating HypernetworkModule for size {size} during load: {e}")
                     errors.report(f"Error creating HypernetworkModule for size {size} in {self.name} during load", exc_info=True)
                     # Should we skip this size or raise the error? Skipping for robustness.
                     continue


        # Load remaining metadata from top level of state_dict
        self.name = state_dict.get('name', self.name) # Keep loaded name if available
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)

        # Ensure modules are moved to the correct device and set to eval mode
        self.to(devices.device)
        self.eval()
        if print_extra: print(f"--- Finished loading {self.name} ---")


    def shorthash(self) -> Optional[str]:
        """Generates a short SHA256 hash of the hypernetwork file."""
        if not self.filename or not os.path.exists(self.filename):
             # Try to find filename from shared dict if not loaded directly
             if self.name and self.name in shared.hypernetworks:
                  self.filename = shared.hypernetworks[self.name]
             else:
                  # print(f"Warning: Cannot generate hash for hypernetwork '{self.name}'. Filename not set or file does not exist.")
                  return None

        # Ensure filename is valid before hashing
        if not self.filename or not isinstance(self.filename, str) or not os.path.exists(self.filename):
             print(f"Warning: Invalid filename '{self.filename}' for hashing hypernetwork '{self.name}'.")
             return None

        try:
            # Use the external hashing function if available
             sha256 = hashes.sha256(self.filename, f'hypernet/{self.name}')
             return sha256[0:10] if sha256 else None
        except Exception as e:
             print(f"Error generating hash for {self.filename}: {e}")
             errors.report(f"Error generating hash for hypernetwork {self.name}", exc_info=True)
             return None


# --- Helper Functions ---

def list_hypernetworks(path: str) -> Dict[str, str]:
    """
    Lists all .pt hypernetwork files found in the given path and subdirectories.

    Args:
        path: The directory path to search for hypernetworks.

    Returns:
        A dictionary mapping hypernetwork names (without extension) to their full file paths.
    """
    res: Dict[str, str] = {}
    if not os.path.isdir(path):
         print(f"Warning: Hypernetwork directory not found or invalid: {path}")
         return res
    try:
        # Use iglob for potentially large directories
        glob_pattern = os.path.join(path, '**', '*.pt')
        for filename in sorted(glob.iglob(glob_pattern, recursive=True), key=str.lower):
             if not os.path.isfile(filename): continue # Skip directories that somehow match .pt
             name = os.path.splitext(os.path.basename(filename))[0]
             # Prevent a hypothetical "None.pt" or empty names from being listed.
             if name and name != "None":
                 if name in res:
                      print(f"Warning: Duplicate hypernetwork name '{name}' found. Overwriting path.")
                 res[name] = filename
    except Exception as e:
        print(f"Error listing hypernetworks in {path}: {e}")
        errors.report(f"Error listing hypernetworks in {path}", exc_info=True)
    return res


def load_hypernetwork(name: str) -> Optional[Hypernetwork]:
    """
    Loads a hypernetwork by name using the path stored in shared.hypernetworks.

    Args:
        name: The name of the hypernetwork to load.

    Returns:
        The loaded Hypernetwork object, or None if loading fails.
    """
    # Check shared object and hypernetworks attribute exist
    if not hasattr(shared, 'hypernetworks') or not isinstance(shared.hypernetworks, dict):
         print(f"Error: shared.hypernetworks not found or not a dictionary.")
         return None

    path = shared.hypernetworks.get(name, None)

    if path is None:
        print(f"Hypernetwork '{name}' not found in shared list.")
        return None
    if not isinstance(path, str) or not os.path.exists(path):
        print(f"Hypernetwork file path is invalid or file not found for '{name}': {path}")
        # Optionally remove invalid entry? shared.hypernetworks.pop(name, None)
        return None

    try:
        # Instantiate with the name, let load overwrite if name is in the file
        hypernetwork = Hypernetwork(name=name)
        hypernetwork.load(path)
        # Device placement and eval mode are handled within hypernetwork.load() now
        return hypernetwork
    except FileNotFoundError:
        print(f"Error: Hypernetwork file not found at path: {path}")
        errors.report(f"FileNotFoundError loading hypernetwork {path}", exc_info=False) # Don't need full trace for FileNotFoundError
        return None
    except Exception as e:
        # Catch other potential errors during loading/instantiation
        print(f"Error loading hypernetwork {name} from {path}: {e}")
        errors.report(f"Error loading hypernetwork {path}", exc_info=True)
        return None


def load_hypernetworks(names: List[str], multipliers: Optional[List[float]] = None):
    """
    Loads multiple hypernetworks by name into shared.loaded_hypernetworks,
    reusing already loaded ones and applying multipliers.

    Args:
        names: A list of hypernetwork names to load.
        multipliers: An optional list of multipliers corresponding to the names.
                     If provided, length must match names. Defaults to 1.0 for all.
    """
    if not hasattr(shared, 'loaded_hypernetworks') or not isinstance(shared.loaded_hypernetworks, list):
         print("Error: shared.loaded_hypernetworks not found or not a list. Cannot load.")
         return

    if multipliers and len(names) != len(multipliers):
        print("Warning: Number of hypernetwork names and multipliers do not match. Applying multiplier 1.0 to all.")
        multipliers = None

    newly_loaded_networks: List[Hypernetwork] = []
    # Use a map for efficient lookup of currently loaded networks
    current_loaded_map: Dict[str, Hypernetwork] = {hn.name: hn for hn in shared.loaded_hypernetworks if hn.name}

    print(f"Loading hypernetworks: {names}")
    for i, name in enumerate(names):
        if not isinstance(name, str) or not name:
             print(f"Warning: Skipping invalid hypernetwork name at index {i}: {name}")
             continue

        hypernetwork = current_loaded_map.get(name)

        if hypernetwork is None:
            print(f"Attempting to load hypernetwork: {name}")
            hypernetwork = load_hypernetwork(name)
            if hypernetwork is None:
                print(f"Failed to load hypernetwork: {name}. Skipping.")
                # Optionally notify user more prominently?
                continue # Skip if loading failed
        else:
            # If reusing, ensure it's still valid (e.g., file wasn't deleted) - maybe add a check?
            print(f"Reusing already loaded hypernetwork: {name}")

        # Set multiplier for this specific network
        multiplier = multipliers[i] if multipliers is not None else 1.0
        try:
             hypernetwork.set_multiplier(multiplier)
             # Ensure it's on the correct device and in eval mode (load should handle this, but belt & suspenders)
             hypernetwork.to(devices.device)
             hypernetwork.eval()
             print(f"Applied multiplier {multiplier} to {name}")
             newly_loaded_networks.append(hypernetwork)
        except Exception as e:
             print(f"Error applying settings or moving to device for hypernetwork {name}: {e}")
             errors.report(f"Error processing hypernetwork {name} after load/reuse", exc_info=True)
             # Decide whether to skip adding it if settings fail


    # Replace the shared list with the newly processed list
    shared.loaded_hypernetworks.clear()
    shared.loaded_hypernetworks.extend(newly_loaded_networks)
    loaded_names = [hn.name for hn in shared.loaded_hypernetworks]
    print(f"Active hypernetworks: {loaded_names if loaded_names else 'None'}")


def apply_single_hypernetwork(hypernetwork: Hypernetwork,
                             context_k: torch.Tensor,
                             context_v: torch.Tensor,
                             layer: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a single hypernetwork's modules to the Key and Value context tensors.

    Args:
        hypernetwork: The Hypernetwork instance to apply.
        context_k: The Key context tensor (e.g., from CrossAttention). Shape (batch, seq_len, dim).
        context_v: The Value context tensor. Shape (batch, seq_len, dim).
        layer: Optional reference to the attention layer being processed (for potential debugging/inspection).

    Returns:
        A tuple containing the modified Key and Value context tensors.
    """
    if hypernetwork is None or not isinstance(hypernetwork, Hypernetwork):
         print("Warning: apply_single_hypernetwork called with invalid hypernetwork object.")
         return context_k, context_v

    # Determine the dimension size from the context tensor's last dimension
    if context_k.ndim < 3 or context_v.ndim < 3:
        print(f"Warning: Context tensors have unexpected dimensions: K={context_k.shape}, V={context_v.shape}. Skipping hypernetwork {hypernetwork.name}.")
        return context_k, context_v
    context_dim = context_k.shape[-1]
    if context_dim != context_v.shape[-1]:
         print(f"Warning: K ({context_k.shape[-1]}) and V ({context_v.shape[-1]}) context dimensions mismatch. Skipping hypernetwork {hypernetwork.name}.")
         return context_k, context_v

    # Get the corresponding K and V modules for this dimension
    hypernetwork_modules = hypernetwork.layers.get(context_dim)

    if hypernetwork_modules is None:
        # This is common if the HN doesn't support all dimensions in the model
        # print(f"Debug: Hypernetwork '{hypernetwork.name}' has no layers for dimension {context_dim}.")
        return context_k, context_v

    module_k, module_v = hypernetwork_modules

    # Ensure modules are on the correct device and in eval mode
    # This should ideally be guaranteed by load_hypernetworks, but check just in case
    module_k.to(devices.device).eval()
    module_v.to(devices.device).eval()

    # Optional: Store modules in the attention layer for potential inspection/debugging
    # This depends on the structure of the 'layer' object passed in
    if layer is not None:
         try:
             # Use setattr for flexibility, guarded by try-except
             setattr(layer, 'hyper_k', module_k)
             setattr(layer, 'hyper_v', module_v)
         except AttributeError:
              # Ignore if layer doesn't support these attributes
              pass

    # Apply the modules
    # Use devices casting functions for compatibility (e.g., with half-precision)
    try:
         # Ensure inputs are float before module forward pass if needed by modules
         k_float = devices.cond_cast_float(context_k)
         v_float = devices.cond_cast_float(context_v)

         # Apply modules and cast back to unet's expected type
         new_context_k = devices.cond_cast_unet(module_k(k_float))
         new_context_v = devices.cond_cast_unet(module_v(v_float))

         return new_context_k, new_context_v

    except Exception as e:
         print(f"Error applying hypernetwork module from '{hypernetwork.name}' for dim {context_dim}: {e}")
         errors.report(f"Error during hypernetwork forward pass for {hypernetwork.name}, dim {context_dim}", exc_info=True)
         # Return original contexts on error to avoid crashing generation
         return context_k, context_v


def apply_hypernetworks(hypernetworks: List[Hypernetwork],
                       context: torch.Tensor,
                       layer: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a list of hypernetworks sequentially to the context tensor.

    Args:
        hypernetworks: A list of Hypernetwork instances to apply (usually from shared.loaded_hypernetworks).
        context: The input context tensor (shared for K and V initially).
        layer: Optional reference to the attention layer being processed.

    Returns:
        A tuple containing the modified Key and Value context tensors after all hypernetworks are applied.
    """
    if not hypernetworks: # If the list is empty, return original context
        return context, context

    # Start with the original context for both K and V
    context_k = context
    context_v = context

    # Apply each hypernetwork sequentially
    for hypernetwork in hypernetworks:
        if not isinstance(hypernetwork, Hypernetwork): # Sanity check
            print(f"Warning: Invalid object in hypernetworks list: {type(hypernetwork)}. Skipping.")
            continue
        # Apply the single HN. It handles device/eval mode internally now.
        context_k, context_v = apply_single_hypernetwork(hypernetwork, context_k, context_v, layer)

    return context_k, context_v


def attention_CrossAttention_forward(self: Any, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
    """
    MONKEY PATCHED forward method for a CrossAttention layer.
    Injects hypernetwork modifications to K and V contexts before projections.
    Assumes the layer has attributes like 'heads', 'to_q', 'to_k', 'to_v', 'scale', 'to_out'.

    Args:
        self: The instance of the CrossAttention layer.
        x: Input tensor (query precursor). Shape (batch, seq_len, dim).
        context: Context tensor (key/value precursor). Shape (batch, context_seq_len, dim). Optional.
        mask: Optional attention mask.
        **kwargs: Additional keyword arguments.

    Returns:
        The output tensor after attention. Shape (batch, seq_len, dim).
    """
    # --- Preconditions and Attribute Checks ---
    if not all(hasattr(self, attr) for attr in ['heads', 'to_q', 'to_k', 'to_v', 'scale', 'to_out']):
         raise AttributeError("CrossAttention layer instance is missing required attributes for patched forward method.")
    h = self.heads # Number of attention heads
    if h <= 0: raise ValueError("Number of attention heads must be positive.")

    # --- Q Projection ---
    try:
         q = self.to_q(x) # Query projection
    except Exception as e:
         errors.report(f"Error during Q projection in CrossAttention: {e}", exc_info=True)
         raise e # Re-raise after reporting

    # --- Context Preparation and Hypernetwork Injection ---
    context = default(context, x) # Use self context if none provided

    # Apply loaded hypernetworks (if any) to the context before K and V projections
    # Ensure shared.loaded_hypernetworks exists and is a list
    active_hypernetworks = getattr(shared, 'loaded_hypernetworks', [])
    if not isinstance(active_hypernetworks, list):
        print("Warning: shared.loaded_hypernetworks is not a list. Skipping hypernetwork application.")
        active_hypernetworks = []

    context_k, context_v = apply_hypernetworks(active_hypernetworks, context, self)

    # --- K and V Projections ---
    try:
         k = self.to_k(context_k) # Key projection (using potentially modified context)
         v = self.to_v(context_v) # Value projection (using potentially modified context)
    except Exception as e:
         errors.report(f"Error during K/V projection in CrossAttention (potentially after hypernetworks): {e}", exc_info=True)
         raise e # Re-raise after reporting

    # --- Multi-Head Attention Reshaping ---
    try:
         # Reshape for multi-head attention: (batch, seq_len, heads*head_dim) -> (batch*heads, seq_len, head_dim)
         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    except Exception as e:
         errors.report(f"Error reshaping QKV for multi-head attention: {e}", exc_info=True)
         raise e

    # --- Attention Score Calculation ---
    # Use einsum for clarity: (batch*heads, query_seq_len, head_dim) x (batch*heads, key_seq_len, head_dim) -> (batch*heads, query_seq_len, key_seq_len)
    # Ensure proper scaling
    sim = einsum('b i d, b j d -> b i j', q, k) * float(self.scale) # Ensure scale is float

    # --- Masking ---
    if mask is not None:
        try:
            if mask.shape[0] != q.shape[0] // h or mask.ndim < 2: # Basic check: batch size match and min 2 dims
                 print(f"Warning: Mask shape {mask.shape} incompatible with query batch size {q.shape[0]//h}. Ignoring mask.")
            else:
                 mask = rearrange(mask, 'b ... -> b (...)') # Flatten mask spatial dims if any
                 max_neg_value = -torch.finfo(sim.dtype).max # Use dtype of sim
                 # Repeat mask for each head: (batch, key_seq_len) -> (batch*heads, 1, key_seq_len)
                 mask = repeat(mask, 'b j -> (b h) () j', h=h)
                 # Apply mask (fill padding positions with large negative value)
                 sim.masked_fill_(~mask.bool(), max_neg_value) # Ensure mask is boolean
        except Exception as e:
            errors.report(f"Error applying attention mask: {e}", exc_info=True)
            # Continue without mask? Or raise? Raising might be safer.
            raise e

    # --- Attention Weight Calculation & Application ---
    try:
         attn = sim.softmax(dim=-1) # Softmax over key sequence length

         # Apply attention weights to values
         # einsum: (b*h, q_len, k_len) x (b*h, k_len, v_dim) -> (b*h, q_len, v_dim)
         out = einsum('b i j, b j d -> b i d', attn, v)
    except Exception as e:
         errors.report(f"Error during attention calculation/application: {e}", exc_info=True)
         raise e

    # --- Reshape Output & Final Projection ---
    try:
         # Reshape back: (batch*heads, query_seq_len, head_dim) -> (batch, query_seq_len, heads*head_dim)
         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
         # Final output projection
         return self.to_out(out)
    except Exception as e:
         errors.report(f"Error during output reshape/projection: {e}", exc_info=True)
         raise e


def stack_conds(conds: List[torch.Tensor]) -> torch.Tensor:
    """
    Pads and stacks a list of conditioning tensors (typically embeddings)
    to the same sequence length (dim 0). Padding uses the last vector.

    Args:
        conds: A list of tensors, each usually shaped (seq_len, embed_dim).

    Returns:
        A single tensor stacking the padded inputs along a new batch dimension (dim 0).
        Shape (batch_size, max_seq_len, embed_dim).
    """
    if not conds:
        # Return an empty tensor with expected ndim if possible, otherwise raise error?
        # Returning empty tensor might hide issues downstream. Let's be stricter.
        raise ValueError("Input list 'conds' cannot be empty.")

    # Ensure all inputs are tensors
    if not all(isinstance(c, torch.Tensor) for c in conds):
        raise TypeError("All elements in 'conds' must be torch.Tensors.")

    # Handle single item case
    if len(conds) == 1:
        return conds[0].unsqueeze(0) # Add batch dimension

    # Check embedding dimensions match
    embed_dim = conds[0].shape[-1]
    if not all(c.shape[-1] == embed_dim for c in conds):
        raise ValueError("All tensors in 'conds' must have the same embedding dimension (last dim).")

    # Find the maximum sequence length (dim 0)
    try:
        max_len = max(c.shape[0] for c in conds)
    except IndexError:
        raise ValueError("Cannot determine sequence length (dim 0) for tensors in 'conds'.")


    padded_conds = []
    for i, c in enumerate(conds):
        current_len = c.shape[0]
        if current_len == max_len:
            padded_conds.append(c)
        elif current_len < max_len:
            # Pad with the last vector
            padding_len = max_len - current_len
            if current_len == 0: # Handle empty tensor case
                 print(f"Warning: Cond tensor at index {i} is empty. Padding with zeros.")
                 # Need to know dtype and device
                 last_vector = torch.zeros((1, embed_dim), dtype=conds[0].dtype, device=conds[0].device)
            else:
                 last_vector = c[-1:] # Keep dimension using slicing

            # Ensure padding vector is correctly shaped (1, embed_dim)
            if last_vector.shape != (1, embed_dim):
                raise RuntimeError(f"Unexpected shape for last vector padding: {last_vector.shape}")

            padding = last_vector.expand(padding_len, -1) # Use expand for efficiency
            padded_conds.append(torch.cat([c, padding], dim=0))
        else:
             # This case should ideally not happen if max_len is calculated correctly
             raise RuntimeError(f"Tensor at index {i} has length {current_len} > max_len {max_len}.")


    # Stack along the new batch dimension (dim 0)
    try:
         return torch.stack(padded_conds, dim=0)
    except Exception as e:
         print("Error during torch.stack. Padded tensor shapes:")
         for i, p in enumerate(padded_conds): print(f"  {i}: {p.shape}")
         raise e


def format_statistics(data: deque) -> Tuple[str, str]:
    """
    Calculates and formats loss statistics (overall and recent).
    Provides Mean ± Standard Error of the Mean (SEM).

    Args:
        data: A deque containing recent loss values (or other numeric data).

    Returns:
        A tuple containing formatted strings for:
        (total_information, recent_information)
    """
    if not data:
        return "loss: N/A", "recent loss: N/A"

    # Use a stable list copy for calculations
    data_list = list(data)
    n_total = len(data_list)

    if n_total == 0: # Should be caught by initial check, but double-check
         return "loss: N/A", "recent loss: N/A"

    # Overall statistics
    mean_total = mean(data_list)
    std_total = stdev(data_list) if n_total >= 2 else 0.0
    sem_total = std_total / (n_total ** 0.5) if n_total > 0 else 0.0 # Standard Error of the Mean

    # Use Unicode U+00B1 for plus-minus symbol
    total_info = f"loss: {mean_total:.4f} \u00B1 {sem_total:.4f}"

    # Recent statistics (e.g., last 32 steps)
    recent_data = data_list[-32:]
    n_recent = len(recent_data)

    if n_recent == 0:
        recent_info = "recent loss: N/A"
    else:
        mean_recent = mean(recent_data)
        std_recent = stdev(recent_data) if n_recent >= 2 else 0.0
        sem_recent = std_recent / (n_recent ** 0.5) if n_recent > 0 else 0.0
        recent_info = f"recent 32 loss: {mean_recent:.4f} \u00B1 {sem_recent:.4f}"

    return total_info, recent_info


# --- Uncommented Training/Creation Functions ---
# WARNING: These functions depend heavily on the external environment
# (Stable Diffusion Web UI modules like shared, processing, images, dataset)
# and WILL NOT run correctly standalone. They are included here based
# on the request for the "full" code structure.

def create_hypernetwork(
    name: str,
    enable_sizes: List[int],
    overwrite_old: bool,
    layer_structure: Optional[Union[str, List[float]]] = None,
    activation_func: Optional[str] = None,
    weight_init: Optional[str] = None,
    add_layer_norm: bool = False,
    use_dropout: bool = False,
    dropout_structure: Optional[Union[str, List[float]]] = None,
    optional_info: Optional[str] = None
) -> None:
    """
    Creates and saves a new hypernetwork file with the specified parameters.

    WARNING: Requires `shared.cmd_opts.hypernetwork_dir` and `shared.reload_hypernetworks()`
             from the external environment.

    Args:
        name: The desired name for the hypernetwork (will be sanitized).
        enable_sizes: List of dimension sizes the hypernetwork should support.
        overwrite_old: Whether to overwrite an existing file with the same name.
        layer_structure: Layer structure multipliers. Can be a comma-separated string or list. Defaults to [1, 2, 1].
        activation_func: Activation function name.
        weight_init: Weight initialization method name.
        add_layer_norm: Whether to use layer normalization.
        use_dropout: Whether to use dropout (if dropout_structure not provided).
        dropout_structure: Explicit dropout structure. Can be a comma-separated string or list. Overrides use_dropout flag.
        optional_info: Optional description or metadata to save with the hypernetwork.
    """
    print(f"--- Creating Hypernetwork: {name} ---")
    # --- Input Validation and Sanitization ---
    if not name:
        print("Error: Hypernetwork name cannot be empty.")
        errors.report("Hypernetwork creation failed: Name cannot be empty.")
        return

    # Remove potentially problematic characters from name (allow alphanum, ., _, -, space)
    sanitized_name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    if not sanitized_name:
        print(f"Error: Hypernetwork name '{name}' contains no valid characters.")
        errors.report(f"Hypernetwork creation failed: Invalid name '{name}'.")
        return
    if sanitized_name != name:
         print(f"Sanitized hypernetwork name to: '{sanitized_name}'")
         name = sanitized_name


    # Determine hypernetwork directory path (dependency)
    try:
        hypernetwork_dir = shared.cmd_opts.hypernetwork_dir
        if not hypernetwork_dir or not isinstance(hypernetwork_dir, str):
             raise ValueError("shared.cmd_opts.hypernetwork_dir is not configured.")
    except AttributeError:
        print("Error: Could not find hypernetwork directory path in shared.cmd_opts.")
        errors.report("Hypernetwork creation failed: Hypernetwork directory configuration missing.", exc_info=True)
        return

    # Ensure directory exists
    os.makedirs(hypernetwork_dir, exist_ok=True)
    filename = os.path.join(hypernetwork_dir, f"{name}.pt")

    if not overwrite_old and os.path.exists(filename):
        print(f"Error: Hypernetwork file {filename} already exists and overwrite is disabled.")
        errors.report(f"Hypernetwork creation failed: File '{filename}' already exists.")
        return

    # Parse layer structure string if needed
    parsed_layer_structure: List[Union[int, float]] # Type hint for clarity
    if layer_structure is None:
         parsed_layer_structure = [1.0, 2.0, 1.0] # Default
    elif isinstance(layer_structure, str):
        try:
            parsed_layer_structure = [float(x.strip()) for x in layer_structure.split(",") if x.strip()]
            if not parsed_layer_structure or parsed_layer_structure[0] != 1 or parsed_layer_structure[-1] != 1:
                 raise ValueError("Layer structure must start and end with 1.")
        except ValueError as e:
            print(f"Error: Invalid layer structure string '{layer_structure}': {e}")
            errors.report(f"Hypernetwork creation failed: Invalid layer structure '{layer_structure}'.", exc_info=True)
            return
    elif isinstance(layer_structure, list) and all(isinstance(x, (int, float)) for x in layer_structure):
         parsed_layer_structure = layer_structure
         if not parsed_layer_structure or parsed_layer_structure[0] != 1 or parsed_layer_structure[-1] != 1:
             print("Error: Invalid layer structure list. Must start and end with 1.")
             errors.report(f"Hypernetwork creation failed: Invalid layer structure list.")
             return
    else:
        print(f"Error: Invalid type for layer structure: {type(layer_structure)}. Expected list or string.")
        errors.report(f"Hypernetwork creation failed: Invalid layer structure type.")
        return


    # Parse dropout structure string if needed
    parsed_dropout_structure: Optional[List[float]] = None # Explicit structure takes precedence
    if isinstance(dropout_structure, str):
        try:
            parsed_dropout_structure = [float(x.strip()) for x in dropout_structure.split(",") if x.strip()]
            if len(parsed_dropout_structure) != len(parsed_layer_structure):
                 raise ValueError("Dropout structure length must match layer structure length.")
            if not all(0.0 <= p < 1.0 for p in parsed_dropout_structure):
                 raise ValueError("Dropout probabilities must be between 0.0 (inclusive) and 1.0 (exclusive).")
            # If structure is provided, use_dropout flag becomes informational based on content
            use_dropout = any(p > 0 for p in parsed_dropout_structure)
        except ValueError as e:
            print(f"Error: Invalid dropout structure string '{dropout_structure}': {e}")
            errors.report(f"Hypernetwork creation failed: Invalid dropout structure '{dropout_structure}'.", exc_info=True)
            return
    elif isinstance(dropout_structure, list) and all(isinstance(x, float) for x in dropout_structure):
         if len(dropout_structure) != len(parsed_layer_structure):
             print("Error: Dropout structure list length must match layer structure list length.")
             errors.report("Hypernetwork creation failed: Dropout structure length mismatch.")
             return
         if not all(0.0 <= p < 1.0 for p in dropout_structure):
             print("Error: Dropout probabilities must be between 0.0 (inclusive) and 1.0 (exclusive).")
             errors.report("Hypernetwork creation failed: Invalid dropout probability value.")
             return
         parsed_dropout_structure = dropout_structure
         use_dropout = any(p > 0 for p in parsed_dropout_structure)
    elif dropout_structure is not None:
         print(f"Error: Invalid type for dropout structure: {type(dropout_structure)}. Expected list or string.")
         errors.report(f"Hypernetwork creation failed: Invalid dropout structure type.")
         return
    # else: parsed_dropout_structure remains None, will be generated by Hypernetwork constructor


    # Validate enable_sizes
    if not enable_sizes or not isinstance(enable_sizes, list) or not all(isinstance(x, int) and x > 0 for x in enable_sizes):
        print(f"Error: Invalid 'enable_sizes': {enable_sizes}. Must be a list of positive integers.")
        errors.report(f"Hypernetwork creation failed: Invalid enable_sizes.")
        return

    # --- Hypernetwork Instantiation and Saving ---
    try:
        print("Instantiating Hypernetwork object...")
        # Pass parsed structures and flags
        hypernet = Hypernetwork(
            name=name,
            enable_sizes=enable_sizes,
            layer_structure=parsed_layer_structure,
            activation_func=activation_func,
            weight_init=weight_init, # Defaults to 'Normal' in constructor if None
            add_layer_norm=add_layer_norm,
            use_dropout=use_dropout, # Pass the flag for potential structure generation
            dropout_structure=parsed_dropout_structure, # Pass explicit structure if parsed
            optional_info=optional_info
        )
        print(f"Saving hypernetwork to {filename}...")
        hypernet.save(filename) # Save method now handles errors internally
        print(f"Hypernetwork '{name}' created successfully.")

    except Exception as e:
        print(f"Error during hypernetwork instantiation or saving: {e}")
        errors.report(f"Hypernetwork creation failed for '{name}'", exc_info=True)
        # Clean up potentially partially saved file?
        if os.path.exists(filename):
             try: os.remove(filename)
             except OSError: pass
        return

    # --- Reload Hypernetworks List (Dependency) ---
    try:
        print("Reloading hypernetworks list...")
        shared.reload_hypernetworks()
    except AttributeError:
        print("Warning: shared.reload_hypernetworks() function not found. Cannot reload list.")
    except Exception as e:
        print(f"Error reloading hypernetworks list: {e}")
        errors.report(f"Error reloading hypernetworks list after creating {name}", exc_info=True)

    print(f"--- Finished Creating Hypernetwork: {name} ---")


def train_hypernetwork(
    id_task: str, # Task ID from UI?
    hypernetwork_name: str,
    learn_rate: float,
    batch_size: int,
    gradient_step: int, # Gradient accumulation steps
    data_root: str,
    log_directory: str,
    training_width: int,
    training_height: int,
    varsize: bool, # Variable aspect ratio training?
    steps: int, # Max training steps
    clip_grad_mode: str, # 'value', 'norm', or other/None
    clip_grad_value: float, # Clipping threshold
    shuffle_tags: bool,
    tag_drop_out: float, # Dropout probability for tags (0-1)
    latent_sampling_method: str, # 'once', 'deterministic', 'random'
    use_weight: bool, # Use dataset weights?
    create_image_every: Optional[int], # Step interval for preview images (or None)
    save_hypernetwork_every: Optional[int], # Step interval for saving checkpoints (or None)
    template_filename: str, # Template file for captions
    preview_from_txt2img: bool, # Use txt2img settings for preview?
    preview_prompt: str,
    preview_negative_prompt: str,
    preview_steps: int,
    preview_sampler_name: str,
    preview_cfg_scale: float,
    preview_seed: int,
    preview_width: int,
    preview_height: int
) -> Tuple[Optional[Hypernetwork], Optional[str]]:
    """
    Trains a specified hypernetwork based on provided parameters and dataset.

    WARNING: This function is extremely dependent on the external environment
             (Stable Diffusion Web UI) including `shared`, `processing`, `images`,
             `dataset`, `sd_models`, `devices`, `textual_inversion`, etc.
             It WILL NOT run correctly standalone.

    Args:
        id_task: Identifier for the training task.
        hypernetwork_name: Name of the hypernetwork to train.
        learn_rate: Learning rate for the optimizer.
        batch_size: Number of images per batch.
        gradient_step: Number of batches for gradient accumulation.
        data_root: Path to the dataset directory.
        log_directory: Path to the directory for logs, checkpoints, and previews.
        training_width: Target width for training images/latents.
        training_height: Target height for training images/latents.
        varsize: Whether to handle variable aspect ratios in the dataset.
        steps: Maximum number of training steps to perform.
        clip_grad_mode: Gradient clipping mode ('value', 'norm', or None).
        clip_grad_value: Threshold for gradient clipping.
        shuffle_tags: Whether to shuffle caption tags during training.
        tag_drop_out: Probability of dropping tags from captions (0 to 1).
        latent_sampling_method: How VAE latents are sampled ('once', 'deterministic', 'random').
        use_weight: Whether to use 'weight' column from dataset captions for loss weighting.
        create_image_every: Interval (in steps) to generate preview images. 0 or None disables it.
        save_hypernetwork_every: Interval (in steps) to save hypernetwork checkpoints. 0 or None disables it.
        template_filename: Name of the caption template file.
        preview_from_txt2img: Use txt2img settings for previews instead of dataset prompts.
        preview_prompt: Prompt for txt2img previews.
        preview_negative_prompt: Negative prompt for txt2img previews.
        preview_steps: Steps for txt2img previews.
        preview_sampler_name: Sampler name for txt2img previews.
        preview_cfg_scale: CFG scale for txt2img previews.
        preview_seed: Seed for txt2img previews.
        preview_width: Width for txt2img previews.
        preview_height: Height for txt2img previews.

    Returns:
        A tuple containing:
        - The trained Hypernetwork object (or None on failure).
        - The path to the final saved hypernetwork file (or None on failure).
    """
    # --- Initial Setup & Validation ---
    print(f"--- Starting Hypernetwork Training: {hypernetwork_name} ---")
    print(f"Task ID: {id_task}")

    # Ensure intervals are non-negative integers
    save_hypernetwork_every = max(0, save_hypernetwork_every or 0)
    create_image_every = max(0, create_image_every or 0)

    # Validate essential inputs early
    if not hypernetwork_name: errors.report("Training Error: Hypernetwork name is missing."); return None, None
    if not data_root or not os.path.isdir(data_root): errors.report(f"Training Error: Dataset directory '{data_root}' not found."); return None, None
    if not log_directory: errors.report("Training Error: Log directory is missing."); return None, None
    if steps <= 0: errors.report("Training Error: Max steps must be positive."); return None, None

    # --- Environment Setup (External Dependencies) ---
    # These rely heavily on the host environment (e.g., SD Web UI)
    try:
        # Get template file path
        template_info = textual_inversion.textual_inversion_templates.get(template_filename, None)
        if template_info is None or not hasattr(template_info, 'path'):
            errors.report(f"Training Error: Template file '{template_filename}' not found or invalid."); return None, None
        template_file_path = template_info.path
        if not os.path.exists(template_file_path):
             errors.report(f"Training Error: Template file path '{template_file_path}' does not exist."); return None, None

        # Validate inputs using external function
        # This function likely checks ranges, existence etc. within the Web UI context
        textual_inversion.validate_train_inputs(
             hypernetwork_name, learn_rate, batch_size, gradient_step, data_root,
             template_info, template_filename, steps, save_hypernetwork_every,
             create_image_every, log_directory, name="hypernetwork" # Specify type for validation messages
         )

        # Load the target hypernetwork
        path = shared.hypernetworks.get(hypernetwork_name, None)
        if path is None or not os.path.exists(path):
            errors.report(f"Training Error: Hypernetwork '{hypernetwork_name}' not found at path: {path}"); return None, None

        print(f"Loading hypernetwork from: {path}")
        hypernetwork = Hypernetwork() # Instantiate fresh
        hypernetwork.load(path) # Load state
        # Set this as the only active hypernetwork for training
        shared.loaded_hypernetworks = [hypernetwork]

        # Set training state in shared object
        shared.state.job = f"train-hypernetwork: {hypernetwork.name}"
        shared.state.textinfo = "Initializing hypernetwork training..."
        shared.state.job_count = steps
        shared.state.job_no = 0
        shared.state.interrupted = False # Reset interruption flag

        # Prepare filenames and directories
        # Use the name loaded from HN file if different from input arg? Let's use loaded name.
        loaded_hypernet_name = hypernetwork.name or hypernetwork_name.rsplit('(', 1)[0] # Fallback to input arg parsing
        final_filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{loaded_hypernet_name}.pt')

        # Create log directory structure (YYYY-MM-DD/hypernet_name)
        log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), loaded_hypernet_name)
        os.makedirs(log_directory, exist_ok=True)

        hypernetwork_dir = None
        if save_hypernetwork_every > 0:
            hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
            os.makedirs(hypernetwork_dir, exist_ok=True)
            print(f"Checkpoints will be saved to: {hypernetwork_dir}")

        images_dir = None
        if create_image_every > 0:
            images_dir = os.path.join(log_directory, "images")
            os.makedirs(images_dir, exist_ok=True)
            print(f"Preview images will be saved to: {images_dir}")

        # Get current SD checkpoint info (dependency)
        checkpoint = sd_models.select_checkpoint()
        if checkpoint is None:
             errors.report("Training Error: Could not select current SD checkpoint."); return None, None
        print(f"Training using SD Checkpoint: {checkpoint.model_name} ({checkpoint.shorthash})")

        # Check if already trained
        initial_step = hypernetwork.step or 0
        if initial_step >= steps:
            shared.state.textinfo = f"Hypernetwork already trained to step {initial_step} (max steps: {steps})."
            print(shared.state.textinfo)
            return hypernetwork, final_filename # Return loaded HN and expected final path

        # Setup learning rate scheduler
        # Assumes LearnRateScheduler class exists and is imported
        scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
        print(f"Using LearnRateScheduler: rate={learn_rate}, steps={steps}, initial={initial_step}")

        # Setup gradient clipping
        clip_grad = None
        clip_grad_sched = None
        if clip_grad_mode == "value":
             clip_grad = torch.nn.utils.clip_grad_value_
             print(f"Using gradient clipping by value: {clip_grad_value}")
        elif clip_grad_mode == "norm":
             clip_grad = torch.nn.utils.clip_grad_norm_
             print(f"Using gradient clipping by norm: {clip_grad_value}")
        if clip_grad and clip_grad_value > 0:
             clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
             print("Using scheduled gradient clipping value.")
        elif clip_grad:
             print("Warning: Gradient clipping mode selected but value is zero or invalid. Disabling clipping.")
             clip_grad = None


        # Tensorboard setup (optional)
        tensorboard_writer = None
        if shared.opts.training_enable_tensorboard:
            print("Setting up Tensorboard logging...")
            tensorboard_writer = textual_inversion.tensorboard_setup(log_directory)

    except Exception as e:
        errors.report(f"Error during training initialization: {e}", exc_info=True)
        return None, None # Cannot proceed if setup fails

    # --- Dataset Preparation (External Dependency) ---
    try:
        shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
        print(shared.state.textinfo)
        pin_memory = shared.opts.pin_memory

        # Instantiate dataset and dataloader (complex external classes)
        ds = dataset.PersonalizedBase(
            data_root=data_root, width=training_width, height=training_height,
            repeats=shared.opts.training_image_repeats_per_epoch,
            placeholder_token=loaded_hypernet_name, # Use the actual HN name
            model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model,
            device=devices.device, template_file=template_file_path,
            include_cond=True, batch_size=batch_size, gradient_step=gradient_step,
            shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out,
            latent_sampling_method=latent_sampling_method,
            varsize=varsize, use_weight=use_weight
        )

        # Save training settings (optional)
        if shared.opts.save_training_settings_to_txt:
             print("Saving training settings...")
             # Gather relevant hypernetwork parameters
             hn_params_to_save = {
                 field: getattr(hypernetwork, field, 'N/A')
                 for field in ['layer_structure', 'activation_func', 'weight_init',
                               'add_layer_norm', 'use_dropout', 'dropout_structure',
                               'activate_output']
             }
             # Gather local variables dynamically (be careful with scope)
             # Use a predefined list or locals().copy() filtered
             settings_to_save = {
                 # Core settings
                 'hypernetwork_name': loaded_hypernet_name, 'learn_rate': learn_rate, 'batch_size': batch_size,
                 'gradient_accumulation_steps': gradient_step, 'dataset_directory': data_root, 'log_directory': log_directory,
                 'training_width': training_width, 'training_height': training_height, 'max_training_steps': steps,
                 # Dataset/Augmentation
                 'variable_size': varsize, 'shuffle_tags': shuffle_tags, 'tag_dropout_rate': tag_drop_out,
                 'latent_sampling_method': latent_sampling_method, 'use_dataset_weights': use_weight,
                 'caption_template_file': template_filename,
                 # Optimization
                 'optimizer_name': hypernetwork.optimizer_name or 'Default (AdamW)', # Show which optimizer will be used
                 'gradient_clipping_mode': clip_grad_mode, 'gradient_clipping_value': clip_grad_value,
                 # Checkpoint/Preview
                 'save_checkpoint_every': save_hypernetwork_every, 'create_preview_every': create_image_every,
                 # Associated models/hashes
                 'sd_model_name': checkpoint.model_name, 'sd_model_hash': checkpoint.shorthash,
                 'hypernetwork_step_at_start': initial_step, 'num_dataset_images': len(ds) if ds else 'N/A',
                 # Hypernetwork structure details
                 **hn_params_to_save,
                 # Preview settings
                 'preview_uses_txt2img_settings': preview_from_txt2img,
                 'preview_prompt': preview_prompt, 'preview_negative_prompt': preview_negative_prompt,
                 'preview_steps': preview_steps, 'preview_sampler': preview_sampler_name,
                 'preview_cfg_scale': preview_cfg_scale, 'preview_seed': preview_seed,
                 'preview_width': preview_width, 'preview_height': preview_height
             }
             saving_settings.save_settings_to_file(log_directory, settings_to_save)


        actual_batch_size = ds.batch_size
        actual_gradient_step = ds.gradient_step
        if not actual_batch_size or not actual_gradient_step:
             raise ValueError("Dataset did not return valid batch size or gradient step.")

        # Use the dataloader provided by the external module
        dl = dataset.PersonalizedDataLoader(ds, latent_sampling_method=ds.latent_sampling_method, batch_size=actual_batch_size, pin_memory=pin_memory)
        print(f"Dataset prepared. Size: {len(ds)}, Batch Size: {actual_batch_size}, Grad Accum: {actual_gradient_step}")

    except Exception as e:
        errors.report(f"Error preparing dataset or dataloader: {e}", exc_info=True)
        shared.state.textinfo = f"Error preparing dataset: {e}"
        return None, None

    # --- Model and Optimizer Setup ---
    try:
        # Handle model unloading if configured
        unload = shared.opts.unload_models_when_training
        old_parallel_processing_allowed = shared.parallel_processing_allowed # type: ignore # Assume exists

        if unload:
            print("Unloading SD models to CPU for training...")
            shared.parallel_processing_allowed = False # type: ignore
            shared.sd_model.cond_stage_model.to(devices.cpu)
            shared.sd_model.first_stage_model.to(devices.cpu)
            torch.cuda.empty_cache() # Clear cache after moving

        # Get hypernetwork weights
        weights = hypernetwork.weights()
        if not weights:
             raise RuntimeError("Hypernetwork has no trainable weights.")

        # Set hypernetwork to training mode
        hypernetwork.train()

        # Initialize Optimizer
        optimizer_name = hypernetwork.optimizer_name or 'AdamW' # Use loaded name or default
        if optimizer_name in optimizer_dict:
            optimizer = optimizer_dict[optimizer_name](params=weights, lr=scheduler.learn_rate)
            print(f"Using optimizer: {optimizer_name}")
        else:
            print(f"Warning: Optimizer type '{optimizer_name}' not found in registry! Defaulting to AdamW.")
            optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate)
            optimizer_name = 'AdamW' # Update the name to reflect the actual optimizer used

        # Load optimizer state if available (and name matches - important!)
        if hypernetwork.optimizer_state_dict and hypernetwork.optimizer_name == optimizer_name:
            try:
                optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
                print("Loaded optimizer state from hypernetwork.")
                # Optionally clear the loaded state from hypernetwork object to save memory?
                # hypernetwork.optimizer_state_dict = None # Do this *after* training finishes
            except RuntimeError as e:
                print(f"Warning: Could not resume optimizer state ({optimizer_name}): {e}. Starting fresh.")
                # Potentially clear invalid state?
                hypernetwork.optimizer_state_dict = None
        elif hypernetwork.optimizer_state_dict and hypernetwork.optimizer_name != optimizer_name:
             print(f"Warning: Saved optimizer type ({hypernetwork.optimizer_name}) differs from requested ({optimizer_name}). Ignoring saved state.")
             hypernetwork.optimizer_state_dict = None


        # Initialize GradScaler for AMP (Automatic Mixed Precision)
        scaler = GradScaler()
        print("Optimizer and GradScaler initialized.")

    except Exception as e:
        errors.report(f"Error setting up optimizer or models: {e}", exc_info=True)
        shared.state.textinfo = f"Error setting up optimizer: {e}"
        # Restore models if unloaded?
        if unload:
             shared.sd_model.cond_stage_model.to(devices.device)
             shared.sd_model.first_stage_model.to(devices.device)
             shared.parallel_processing_allowed = old_parallel_processing_allowed
        return None, None


    # --- Training Loop ---
    # Calculate steps per epoch for informational purposes
    # Use actual batch size and grad step from dataset object
    steps_per_epoch = (len(ds) // actual_batch_size // actual_gradient_step) if actual_batch_size * actual_gradient_step > 0 else 0
    # Calculate max steps usable per epoch with gradient accumulation
    max_steps_this_epoch = (len(ds) // actual_batch_size) if actual_batch_size > 0 else 0
    max_accumulation_steps = max_steps_this_epoch - (max_steps_this_epoch % actual_gradient_step) if actual_gradient_step > 0 else max_steps_this_epoch

    loss_step = 0.0      # Loss for the current step (after accumulation)
    _loss_accum = 0.0    # Accumulated loss over gradient steps
    # Configure loss logging deque size (e.g., 3 epochs worth of steps)
    loss_logging_maxlen = (steps_per_epoch * 3) if steps_per_epoch > 0 else 100
    loss_logging: deque = deque(maxlen=loss_logging_maxlen)

    last_saved_file = "<none>"
    last_saved_image = "<none>"

    # Use tqdm for progress bar
    pbar = tqdm.tqdm(total=steps - initial_step, desc=f"Training HN '{loaded_hypernet_name}'")
    shared.state.job_no = initial_step # Update job progress

    # Ensure SD Hijack is active for hypernetwork application
    sd_hijack_checkpoint.add()

    try:
        # Loop for the required number of steps (or dataset iterations)
        # Need to handle epochs vs steps carefully. The logic iterates through the dataloader.
        # It seems designed to stop based on `hypernetwork.step` reaching `steps`.
        # We might need multiple passes through the dataloader (epochs).

        steps_done_in_loop = 0 # Track steps completed within this run
        while hypernetwork.step < steps:

            # Check for interruption at start of each potential epoch
            if shared.state.interrupted:
                 print("Training interrupted.")
                 break

            # Iterate through dataset
            for i, batch in enumerate(dl):
                 # Check for interruption before processing batch
                 if shared.state.interrupted: break
                 # Stop if max steps reached
                 if hypernetwork.step >= steps: break

                 # Apply LR schedule *before* optimizer step
                 scheduler.apply(optimizer, hypernetwork.step)
                 if scheduler.finished: # Should align with step check, but good practice
                      print("Learning rate scheduler finished.")
                      break

                 # Apply gradient clipping schedule if active
                 if clip_grad and clip_grad_sched:
                      clip_grad_sched.step(hypernetwork.step)
                      current_clip_value = clip_grad_sched.learn_rate # Use scheduled value
                 elif clip_grad:
                      current_clip_value = clip_grad_value # Use fixed value
                 else:
                      current_clip_value = None # No clipping

                 # --- Forward Pass ---
                 with devices.autocast(): # Enable AMP
                     # Move batch data to device (non_blocking if pin_memory=True)
                     latents = batch.latent_sample.to(devices.device, non_blocking=pin_memory)

                     # Get conditionings (handle tag shuffle/dropout within dataset/dataloader ideally)
                     # Original code had complex logic here involving moving cond_model to device. Assume handled by stack_conds or dataset.
                     if hasattr(batch, 'cond') and batch.cond:
                         cond = stack_conds(batch.cond).to(devices.device, non_blocking=pin_memory)
                     elif hasattr(batch, 'cond_text') and batch.cond_text:
                         # This implies on-the-fly encoding needed - expensive!
                         print("Warning: On-the-fly text encoding during training loop detected. This is inefficient.")
                         with torch.no_grad(): # Ensure model is on correct device for encoding
                              shared.sd_model.cond_stage_model.to(devices.device)
                              cond = shared.sd_model.cond_stage_model(batch.cond_text).to(devices.device, non_blocking=pin_memory)
                              if unload: shared.sd_model.cond_stage_model.to(devices.cpu) # Move back if unloading
                     else:
                         raise RuntimeError("Batch has no valid conditioning information ('cond' or 'cond_text').")


                     # Calculate loss (potentially weighted)
                     if use_weight and hasattr(batch, 'weight'):
                         weights_tensor = batch.weight.to(devices.device, non_blocking=pin_memory)
                         loss = shared.sd_model.weighted_forward(latents, cond, weights_tensor)[0]
                     else:
                         loss = shared.sd_model.forward(latents, cond)[0]

                     # Scale loss for gradient accumulation
                     loss_scaled = loss / actual_gradient_step

                 # --- Backward Pass & Gradient Accumulation ---
                 # Use GradScaler for stable backward pass with AMP
                 scaler.scale(loss_scaled).backward()
                 _loss_accum += loss.item() # Accumulate unscaled loss for logging accuracy

                 # Check if we need to perform optimizer step
                 # (i + 1) is the number of batches processed in this epoch pass
                 # Check if it's a multiple of gradient_step or if it's the last usable batch
                 is_last_batch = (i + 1) == max_accumulation_steps
                 is_accumulation_step = (i + 1) % actual_gradient_step == 0

                 if is_accumulation_step or is_last_batch:
                      # Accumulation finished or last batch, step the optimizer

                      # Log accumulated loss (average over accumulation steps)
                      loss_step = _loss_accum / actual_gradient_step
                      loss_logging.append(loss_step)
                      _loss_accum = 0.0 # Reset accumulator

                      # Optional Gradient Clipping (applied to hypernetwork weights)
                      if clip_grad and current_clip_value is not None:
                           # Unscale gradients before clipping
                           scaler.unscale_(optimizer)
                           clip_grad(hypernetwork.weights(), current_clip_value) # Apply clipping

                      # Optimizer Step
                      scaler.step(optimizer)
                      # Update GradScaler scale for next iteration
                      scaler.update()
                      # Zero gradients
                      optimizer.zero_grad(set_to_none=True) # More efficient

                      # --- Step Update & Logging ---
                      hypernetwork.step += 1
                      steps_done_in_loop += 1
                      pbar.update(1)
                      shared.state.job_no = hypernetwork.step # Update global step counter

                      # Update progress bar description with current loss and epoch info
                      epoch_num = hypernetwork.step // steps_per_epoch if steps_per_epoch > 0 else 0
                      epoch_step = hypernetwork.step % steps_per_epoch if steps_per_epoch > 0 else hypernetwork.step
                      pbar_desc = f"[E {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.5f}"
                      pbar.set_description(pbar_desc)

                      # Log loss to file and Tensorboard
                      if loss_logging: # Check if deque has data
                           _, recent_loss_info = format_statistics(loss_logging)
                           textual_inversion.write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, steps_per_epoch, {
                               "loss": f"{loss_step:.7f}", # Log instantaneous step loss
                               "learn_rate": scheduler.learn_rate # Log current LR
                           })
                           if tensorboard_writer:
                                # Log smoothed loss to Tensorboard
                                mean_loss = sum(loss_logging) / len(loss_logging)
                                textual_inversion.tensorboard_add(
                                     tensorboard_writer, loss=mean_loss, global_step=hypernetwork.step,
                                     step=epoch_step + 1, # Use 1-based step within epoch
                                     learn_rate=scheduler.learn_rate, epoch_num=epoch_num
                                 )

                      # --- Save Checkpoint ---
                      if hypernetwork_dir is not None and save_hypernetwork_every > 0 and hypernetwork.step % save_hypernetwork_every == 0:
                           checkpoint_name = f'{loaded_hypernet_name}-{hypernetwork.step}'
                           last_saved_file = os.path.join(hypernetwork_dir, f'{checkpoint_name}.pt')
                           print(f"\nSaving checkpoint: {last_saved_file}...")
                           # Assign optimizer state before saving helper is called
                           hypernetwork.optimizer_name = optimizer_name
                           if shared.opts.save_optimizer_state:
                               hypernetwork.optimizer_state_dict = optimizer.state_dict()
                           else:
                               hypernetwork.optimizer_state_dict = None # Ensure it's None if not saving

                           save_hypernetwork(hypernetwork, checkpoint, checkpoint_name, last_saved_file)
                           # Keep optimizer state in hypernetwork object until final save
                           # hypernetwork.optimizer_state_dict = None # Don't clear yet

                      # --- Generate Preview Image ---
                      if images_dir is not None and create_image_every > 0 and hypernetwork.step % create_image_every == 0:
                          print(f"\nGenerating preview image for step {hypernetwork.step}...")
                          # Need to put hypernetwork in eval mode and manage RNG state
                          current_mode_is_training = hypernetwork.training # Remember current mode
                          hypernetwork.eval() # Set HN to eval
                          # Save RNG states
                          rng_state = torch.get_rng_state()
                          cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                          # Ensure relevant models are on device
                          shared.sd_model.cond_stage_model.to(devices.device)
                          shared.sd_model.first_stage_model.to(devices.device)

                          # Setup processing object
                          p = processing.StableDiffusionProcessingTxt2Img( # type: ignore
                              sd_model=shared.sd_model,
                              do_not_save_grid=True,
                              do_not_save_samples=True,
                          )
                          p.disable_extra_networks = True # Ensure only the trained HN is active

                          if preview_from_txt2img:
                              p.prompt = preview_prompt
                              p.negative_prompt = preview_negative_prompt
                              p.steps = preview_steps
                              # Get sampler index/name carefully
                              sampler_name = sd_samplers.samplers_map.get(preview_sampler_name.lower(), None) # type: ignore
                              if sampler_name:
                                  p.sampler_name = sampler_name # type: ignore
                              else:
                                   print(f"Warning: Preview sampler '{preview_sampler_name}' not found. Using default.")
                              p.cfg_scale = preview_cfg_scale
                              p.seed = preview_seed
                              p.width = preview_width
                              p.height = preview_height
                          else:
                              # Use a prompt from the current batch if available
                              try:
                                  p.prompt = batch.cond_text[0] # Use first prompt from batch
                              except (AttributeError, IndexError):
                                  print("Warning: Could not get prompt from batch for preview. Using default.")
                                  p.prompt = "preview" # Fallback prompt
                              p.steps = 20 # Default steps
                              p.width = training_width # Use training dimensions
                              p.height = training_height

                          preview_text = p.prompt # Store the prompt used

                          try:
                              # Generate image within closing context for safety
                              with closing(p):
                                  processed = processing.process_images(p) # type: ignore
                                  image = processed.images[0] if processed and processed.images else None

                              if image is not None:
                                  forced_filename = f'{loaded_hypernet_name}-{hypernetwork.step}'
                                  print(f"Saving preview image: {forced_filename}...")
                                  # Save image using external helper
                                  saved_path, saved_info = images.save_image( # type: ignore
                                       image, images_dir, "", p.seed, p.prompt,
                                       shared.opts.samples_format, processed.infotexts[0], p=p,
                                       forced_filename=forced_filename, save_to_dirs=False
                                   )
                                  last_saved_image = f"{saved_path}, prompt: {html.escape(preview_text)}"

                                  # Add image to Tensorboard (optional)
                                  if tensorboard_writer and shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:
                                       # Assume tensorboard_add_image handles PIL Image or Tensor
                                       textual_inversion.tensorboard_add_image(
                                           tensorboard_writer, f"Preview at step {hypernetwork.step}",
                                           image, hypernetwork.step
                                       )
                                  # Assign to shared state for UI update
                                  shared.state.assign_current_image(image)
                              else:
                                   print("Preview image generation failed.")
                                   last_saved_image = "<image generation failed>"

                          except Exception as img_err:
                              print(f"Error generating preview image: {img_err}")
                              errors.report("Error generating preview image during training", exc_info=True)
                              last_saved_image = "<image generation error>"
                          finally:
                              # Restore models to CPU if unloaded
                              if unload:
                                  shared.sd_model.cond_stage_model.to(devices.cpu)
                                  shared.sd_model.first_stage_model.to(devices.cpu)
                              # Restore RNG states
                              torch.set_rng_state(rng_state)
                              if cuda_rng_state: torch.cuda.set_rng_state_all(cuda_rng_state)
                              # Restore hypernetwork training mode
                              if current_mode_is_training: hypernetwork.train()
                              print("Preview generation finished.")


                 # Update shared state text info (even if not stepping optimizer yet)
                 current_loss_info, recent_loss_info = format_statistics(loss_logging)
                 shared.state.textinfo = f"""
<p>
Step: {hypernetwork.step}/{steps} [{current_loss_info}, {recent_loss_info}]<br>
LR: {scheduler.learn_rate:.2e}<br>
Last Checkpoint: {html.escape(last_saved_file)}<br>
Last Preview: {html.escape(last_saved_image)}<br>
</p>
"""
                 # --- End of Batch Processing ---

            # --- End of Epoch / Dataloader Iteration ---
            if shared.state.interrupted: break # Check interruption after finishing epoch pass
            if hypernetwork.step >= steps: break # Check steps after finishing epoch pass

            print(f"Finished dataloader pass. Current step: {hypernetwork.step}. Restarting dataloader.")
            # Loop continues, iterating dl again until steps are met

    except Exception as train_err:
        print(f"\n--- Training Loop Exception ---")
        errors.report("Exception during hypernetwork training loop", exc_info=True)
        shared.state.textinfo = f"Error during training: {train_err}"
    finally:
        # --- Cleanup ---
        print("\n--- Finalizing Training ---")
        pbar.close()
        # Ensure hypernetwork is in eval mode after training
        hypernetwork.eval()
        # Remove SD Hijack
        sd_hijack_checkpoint.remove()

        # Save final hypernetwork state
        print(f"Saving final hypernetwork to: {final_filename}")
        hypernetwork.optimizer_name = optimizer_name # Ensure correct name is saved
        if shared.opts.save_optimizer_state:
             # Make sure optimizer state is current
             hypernetwork.optimizer_state_dict = optimizer.state_dict() if 'optimizer' in locals() else None
        else:
             hypernetwork.optimizer_state_dict = None

        save_hypernetwork(hypernetwork, checkpoint, loaded_hypernet_name, final_filename)

        # Clear optimizer state from memory after final save
        hypernetwork.optimizer_state_dict = None
        if 'optimizer' in locals(): del optimizer # Delete optimizer object

        # Restore models to device if they were unloaded
        if unload:
            print("Restoring SD models to main device...")
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)
            shared.parallel_processing_allowed = old_parallel_processing_allowed # Restore setting

        # Clear loaded hypernetworks? Or keep the trained one loaded?
        # Original code implies it should be returned, so maybe keep it loaded.
        # shared.loaded_hypernetworks = [] # Optional: clear loaded list

        shared.state.job = "" # Clear job state
        shared.state.job_count = 0
        shared.state.job_no = 0

        print("--- Hypernetwork Training Finished ---")
        return hypernetwork, final_filename


# Helper function (previously commented out)
def save_hypernetwork(
    hypernetwork: Hypernetwork,
    checkpoint: Any, # Type hint for checkpoint object (structure unknown)
    hypernetwork_name: str,
    filename: str
) -> None:
    """
    Helper function to save the hypernetwork state with SD checkpoint info.
    Updates the hypernetwork object's metadata before calling its save method.

    Args:
        hypernetwork: The Hypernetwork object to save.
        checkpoint: The SD checkpoint object/info (needs `shorthash` and `model_name` attributes).
        hypernetwork_name: The base name to save the hypernetwork under (may include step count).
        filename: The full path where the hypernetwork file should be saved.
    """
    # Store old metadata in case saving fails
    old_hypernetwork_name = hypernetwork.name
    old_sd_checkpoint = getattr(hypernetwork, "sd_checkpoint", None)
    old_sd_checkpoint_name = getattr(hypernetwork, "sd_checkpoint_name", None)

    try:
        # Assign current checkpoint info and name to the hypernetwork object
        hypernetwork.sd_checkpoint = getattr(checkpoint, 'shorthash', None)
        hypernetwork.sd_checkpoint_name = getattr(checkpoint, 'model_name', None)
        hypernetwork.name = hypernetwork_name # Use the potentially step-suffixed name

        # Call the hypernetwork's own save method
        hypernetwork.save(filename)

    except Exception as e:
        print(f"Error during save_hypernetwork helper for {filename}: {e}")
        # Restore old metadata on failure
        hypernetwork.name = old_hypernetwork_name
        hypernetwork.sd_checkpoint = old_sd_checkpoint
        hypernetwork.sd_checkpoint_name = old_sd_checkpoint_name
        # Re-raise the exception or report it
        errors.report(f"Failed to save hypernetwork checkpoint {filename}", exc_info=True)
        raise e # Re-raise to signal failure


# --- Example Usage (Conceptual - Requires External Environment) ---
if __name__ == '__main__':
    # This block demonstrates instantiation and basic calls.
    # It WILL NOT RUN fully due to dependencies on 'modules' environment.

    print("\n--- Conceptual Example ---")
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {devices.device}")

    # Example: Create a basic hypernetwork instance (requires sizes)
    try:
        hn = Hypernetwork(
            name="test_hypernet",
            enable_sizes=[320, 640],
            layer_structure=[1, 1.5, 1],
            activation_func="relu",
            weight_init="KaimingNormal",
            add_layer_norm=True,
            use_dropout=True, # Generate dropout structure
            # dropout_structure = [0.0, 0.1, 0.0], # Or provide explicitly
            activate_output=False,
            optional_info="Test hypernetwork created conceptually."
        )
        print(f"Created hypernetwork instance: {hn.name}")
        print(f"Layers for dims: {list(hn.layers.keys())}")
        print(f"Dropout structure: {hn.dropout_structure}")
        print(f"Trainable weights: {len(hn.weights())}")
        print(f"Device of first param: {next(iter(hn.weights()), torch.tensor(0)).device}")

        # Simulate saving and loading
        os.makedirs("./temp_hypernets", exist_ok=True)
        save_path = "./temp_hypernets/test_hypernet_enhanced.pt"
        hn.step = 100
        hn.sd_checkpoint = "dummy_sd_hash"
        hn.sd_checkpoint_name = "dummy_sd_model.ckpt"
        hn.optimizer_name = "AdamW"
        # Simulate optimizer state dict
        hn.optimizer_state_dict = {'param_groups': [],
