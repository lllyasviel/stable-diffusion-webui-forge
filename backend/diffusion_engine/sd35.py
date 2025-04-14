import torch
import logging
from typing import Any, Dict, List, Optional, Type, Tuple, Union
from dataclasses import dataclass, field

# Assuming these imports remain valid and necessary
from huggingface_guess import model_list, HuggingfaceGuessV3
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.vae import VAE
from backend.patcher.unet import UnetPatcher
from backend.text_processing.classic_engine import ClassicTextProcessingEngine
from backend.text_processing.t5_engine import T5TextProcessingEngine
from backend.args import dynamic_args # Keep for now, but ideally replaced by config
from backend import memory_management
from backend.modules.k_prediction import PredictionDiscreteFlow, KPredictionBase

# Potentially use shared opts or a dedicated config system
try:
    from modules.shared import opts
except ImportError:
    # Provide fallback defaults if modules.shared isn't available
    class OptsMock:
        sd3_enable_t5 = True
        # Add other necessary opts attributes here
    opts = OptsMock()

logger = logging.getLogger(__name__)

# --- Configuration Dataclass ---
@dataclass
class SD3Config:
    """Configuration specific to the Stable Diffusion 3 engine."""
    embedding_dir: str = field(default_factory=lambda: dynamic_args['embedding_dir'])
    emphasis_name: str = field(default_factory=lambda: dynamic_args['emphasis_name'])
    k_predictor_shift: float = 3.0
    use_t5_encoder: bool = field(default_factory=lambda: getattr(opts, 'sd3_enable_t5', True))
    compile_unet: bool = False # Flag to enable torch.compile on UNet
    compile_vae: bool = False  # Flag to enable torch.compile on VAE
    compile_mode: Optional[str] = None # e.g., "reduce-overhead", "max-autotune"
    # Add more config options as needed: attention optimization flags, quantization settings, etc.

# --- Enhanced ForgeObjects for SD3 ---
class SD3ForgeObjects(ForgeObjects):
    """Specialized ForgeObjects container for SD3 components."""
    # Explicitly type hint components relevant to SD3
    clip_l: Optional[Any] = None
    clip_g: Optional[Any] = None
    t5xxl: Optional[Any] = None
    tokenizer_l: Optional[Any] = None
    tokenizer_g: Optional[Any] = None
    tokenizer_t5: Optional[Any] = None
    transformer: Optional[UnetPatcher] = None # Rename unet -> transformer for clarity

    def __init__(self, transformer: UnetPatcher, vae: VAE, clip: CLIP,
                 clip_l: Any, clip_g: Any, t5xxl: Any,
                 tokenizer_l: Any, tokenizer_g: Any, tokenizer_t5: Any,
                 clipvision: Optional[Any] = None): # clipvision is usually None for SD3 text-to-image
        super().__init__(unet=transformer, clip=clip, vae=vae, clipvision=clipvision) # Keep base happy
        self.transformer = transformer # Explicitly assign transformer
        self.clip_l = clip_l
        self.clip_g = clip_g
        self.t5xxl = t5xxl
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.tokenizer_t5 = tokenizer_t5

    # Override shallow_copy if necessary to handle new attributes correctly
    def shallow_copy(self) -> 'SD3ForgeObjects':
        new_obj = super().shallow_copy() # Get shallow copy from base
        # Ensure SD3 specific attributes are also shallow copied if they are objects
        # For nn.Module references, shallow copy is usually correct.
        new_obj.transformer = self.transformer
        new_obj.clip_l = self.clip_l
        new_obj.clip_g = self.clip_g
        new_obj.t5xxl = self.t5xxl
        new_obj.tokenizer_l = self.tokenizer_l
        new_obj.tokenizer_g = self.tokenizer_g
        new_obj.tokenizer_t5 = self.tokenizer_t5
        return new_obj # Return as SD3ForgeObjects type implicitly

# --- Unified Text Processing Orchestrator ---
class SD3TextProcessingOrchestrator:
    """Handles text processing using CLIP-L, CLIP-G, and T5 encoders."""
    def __init__(self,
                 clip_l_model: Any, tokenizer_l: Any,
                 clip_g_model: Any, tokenizer_g: Any,
                 t5_model: Any, tokenizer_t5: Any,
                 config: SD3Config):
        self.config = config
        self.engine_l = ClassicTextProcessingEngine(
            text_encoder=clip_l_model,
            tokenizer=tokenizer_l,
            embedding_dir=config.embedding_dir,
            embedding_key='clip_l',
            embedding_expected_shape=768,
            emphasis_name=config.emphasis_name,
            text_projection=True, minimal_clip_skip=1, clip_skip=1, # Initial clip_skip
            return_pooled=True, final_layer_norm=False,
        )
        self.engine_g = ClassicTextProcessingEngine(
            text_encoder=clip_g_model,
            tokenizer=tokenizer_g,
            embedding_dir=config.embedding_dir,
            embedding_key='clip_g',
            embedding_expected_shape=1280,
            emphasis_name=config.emphasis_name,
            text_projection=True, minimal_clip_skip=1, clip_skip=1, # Initial clip_skip
            return_pooled=True, final_layer_norm=False,
        )
        self.engine_t5 = T5TextProcessingEngine(
            text_encoder=t5_model,
            tokenizer=tokenizer_t5,
            emphasis_name=config.emphasis_name,
        )
        self.models_on_gpu = False # Track GPU state

    def set_clip_skip(self, clip_skip: int):
        """Sets the CLIP skip value for both L and G encoders."""
        logger.debug(f"Setting CLIP Skip to: {clip_skip}")
        self.engine_l.clip_skip = clip_skip
        self.engine_g.clip_skip = clip_skip

    def load_models_to_gpu(self):
        """Loads necessary text processing models to GPU."""
        if not self.models_on_gpu:
            logger.debug("Loading text processing models to GPU...")
            # Assuming CLIP object handles its own patching/loading
            memory_management.load_model_gpu(self.engine_l.text_encoder)
            memory_management.load_model_gpu(self.engine_g.text_encoder)
            if self.config.use_t5_encoder:
                memory_management.load_model_gpu(self.engine_t5.text_encoder)
            self.models_on_gpu = True
        else:
            logger.debug("Text processing models already on GPU.")

    def process_prompt(self, prompt: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Processes the input prompt(s) through all relevant encoders
        and returns the conditioning dictionary.
        """
        if isinstance(prompt, str):
            prompt = [prompt] # Ensure list format

        self.load_models_to_gpu()

        # Process through individual engines
        cond_g, g_pooled = self.engine_g(prompt)
        cond_l, l_pooled = self.engine_l(prompt)

        if self.config.use_t5_encoder:
            logger.debug("Processing with T5 encoder.")
            cond_t5 = self.engine_t5(prompt)
        else:
            logger.debug("Skipping T5 encoder.")
            # Create appropriately sized zero tensor on the correct device
            batch_size = len(prompt)
            device = cond_l.device
            dtype = cond_l.dtype # Match dtype
            # Determine expected T5 sequence length (often fixed, e.g., 256, check model config)
            # T5 XXL base has sequence length 512 by default, but SD3 might use different settings
            # Using a common value here, adjust if needed based on actual SD3 T5 usage.
            # The original code used 256 seq len. Let's stick to that for compatibility.
            t5_seq_len = 256
            t5_embedding_dim = 4096 # T5 XXL embedding dim
            cond_t5 = torch.zeros([batch_size, t5_seq_len, t5_embedding_dim], device=device, dtype=dtype)

        # Handle potential empty negative prompts
        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)
        force_zero_negative_prompt = is_negative_prompt and all(p == '' for p in prompt)

        if force_zero_negative_prompt:
            logger.debug("Forcing zero tensors for empty negative prompt.")
            l_pooled = torch.zeros_like(l_pooled)
            g_pooled = torch.zeros_like(g_pooled)
            cond_l = torch.zeros_like(cond_l)
            cond_g = torch.zeros_like(cond_g)
            cond_t5 = torch.zeros_like(cond_t5)

        # Combine conditions (Ensure padding matches SD3 expected format)
        # SD3 expects [B, MaxTokens, EmbDim]. CLIP L/G often have 77 tokens. T5 might differ (e.g., 256).
        # Need to ensure correct concatenation and padding strategy.
        # Original code concatenates L and G embeddings and pads to 4096, then concatenates with T5 along seq dim.

        # Concatenate L and G along the embedding dimension
        cond_lg = torch.cat([cond_l, cond_g], dim=-1) # Shape: [B, 77, 768+1280=2048]

        # Pad the combined L+G embedding dimension to match T5's (4096)
        # This seems unusual; typically concatenation happens along sequence length or batch.
        # Replicating original logic here, but review SD3's actual conditioning input spec.
        # Assuming T5 embedding dim is 4096.
        padding_needed = 4096 - cond_lg.shape[-1]
        if padding_needed < 0:
             logger.warning(f"CLIP L+G dim ({cond_lg.shape[-1]}) exceeds target padding dim (4096). Truncating.")
             cond_lg = cond_lg[..., :4096]
        elif padding_needed > 0:
             cond_lg = torch.nn.functional.pad(cond_lg, (0, padding_needed)) # Pad last dim

        # Concatenate padded L+G with T5 along the sequence dimension
        # Shape L+G: [B, 77, 4096], Shape T5: [B, 256, 4096] -> [B, 77+256, 4096] = [B, 333, 4096]
        combined_crossattn = torch.cat([cond_lg, cond_t5], dim=-2)

        # Combine pooled outputs
        combined_vector = torch.cat([l_pooled, g_pooled], dim=-1) # Shape: [B, 768+1280=2048]

        conditioning = dict(
            crossattn=combined_crossattn,
            vector=combined_vector,
        )

        return conditioning

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> Tuple[int, int]:
        """Gets token count using the primary (T5) tokenizer for UI display."""
        # Use T5 tokenizer as it's often the longest/most restrictive
        try:
            tokens = self.engine_t5.tokenizer(prompt, return_tensors="pt").input_ids[0]
            token_count = len(tokens)
            # Determine max length (consult T5 model config or use a reasonable default like 256/512)
            # Using T5-XXL's typical input length, adjust if SD3 uses different.
            max_length = getattr(self.engine_t5.tokenizer, 'model_max_length', 256)
            return token_count, max(max_length - 2, token_count) # Account for potential special tokens
        except Exception as e:
            logger.error(f"Error tokenizing prompt for UI length: {e}")
            return 0, 255 # Fallback


# --- Main Stable Diffusion 3 Engine ---
class StableDiffusion3Enhanced(ForgeDiffusionEngine):
    """
    Enhanced implementation of the ForgeDiffusionEngine for Stable Diffusion 3,
    incorporating improved structure, configuration, performance options, and extensibility.
    """
    # Define which HuggingfaceGuess models this engine can handle
    matched_guesses: List[Type[HuggingfaceGuessV3]] = [model_list.SD3]
    engine_name: str = "StableDiffusion3Enhanced" # Identify the engine

    @classmethod
    def setup_huggingface_heuristic_patches(cls):
        """Applies necessary patches to huggingface_guess heuristics for SD3."""
        logger.info(f"Applying Huggingface heuristics patches for {cls.engine_name}")

        def sd3_clip_target(self, state_dict={}):
             # Maps internal component names to expected keys in HuggingFace model dicts
             return {'clip_l': 'text_encoder', 'clip_g': 'text_encoder_2', 't5xxl': 'text_encoder_3'}

        model_list.SD3.unet_target = 'transformer' # SD3 uses a Transformer, not a UNet
        model_list.SD3.clip_target = sd3_clip_target

    def __init__(self, estimated_config: Dict[str, Any], huggingface_components: Dict[str, Any], config: Optional[SD3Config] = None):
        """
        Initializes the Stable Diffusion 3 Enhanced Engine.

        Args:
            estimated_config: Configuration estimated by HuggingfaceGuess.
            huggingface_components: Dictionary of loaded HuggingFace model components.
            config: Optional SD3Config object for engine-specific settings.
        """
        super().__init__(estimated_config, huggingface_components)
        self.config = config or SD3Config() # Use provided config or defaults
        self.is_inpaint = False # TODO: Add proper inpainting model handling
        self.applied_loras: Dict[str, float] = {} # Track applied LoRAs/adapters

        logger.info(f"Initializing {self.engine_name} with config: {self.config}")

        # --- Component Initialization ---
        try:
            # 1. CLIP/Text Encoders (Managed by CLIP Patcher)
            clip_l = huggingface_components['text_encoder']
            clip_g = huggingface_components['text_encoder_2']
            t5xxl = huggingface_components['text_encoder_3']
            tokenizer_l = huggingface_components['tokenizer']
            tokenizer_g = huggingface_components['tokenizer_2']
            tokenizer_t5 = huggingface_components['tokenizer_3']

            clip = CLIP(
                model_dict={'clip_l': clip_l, 'clip_g': clip_g, 't5xxl': t5xxl},
                tokenizer_dict={'clip_l': tokenizer_l, 'clip_g': tokenizer_g, 't5xxl': tokenizer_t5}
            )

            # 2. VAE
            vae_model = huggingface_components['vae']
            vae = VAE(model=vae_model)
            if self.config.compile_vae:
                logger.info(f"Compiling VAE with mode: {self.config.compile_mode or 'default'}...")
                try:
                    # Ensure VAE methods needed for compile are available or wrap them
                    vae.first_stage_model.encode = torch.compile(vae.first_stage_model.encode, mode=self.config.compile_mode)
                    vae.first_stage_model.decode = torch.compile(vae.first_stage_model.decode, mode=self.config.compile_mode)
                    logger.info("VAE compilation successful (encode/decode).")
                except Exception as e:
                    logger.error(f"VAE compilation failed: {e}", exc_info=True)


            # 3. K-Predictor (Flow Matching specific)
            # TODO: Make KPredictor type configurable
            k_predictor = PredictionDiscreteFlow(shift=self.config.k_predictor_shift)

            # 4. UNet/Transformer Patcher
            transformer_model = huggingface_components['transformer']
            unet_patcher = UnetPatcher.from_model(
                model=transformer_model,
                diffusers_scheduler=None, # SD3 doesn't use diffusers scheduler directly here
                k_predictor=k_predictor,
                config=estimated_config # Pass model's config (channels, etc.)
                # TODO: Add flags for attention optimization backend selection
            )
            if self.config.compile_unet:
                logger.info(f"Compiling Transformer (UNetPatcher) with mode: {self.config.compile_mode or 'default'}...")
                try:
                    # Compile the core model within the patcher
                    unet_patcher.model = torch.compile(unet_patcher.model, mode=self.config.compile_mode, dynamic=True) # Assume dynamic shapes might be needed
                    logger.info("Transformer (UNetPatcher) compilation successful.")
                except Exception as e:
                    logger.error(f"Transformer (UNetPatcher) compilation failed: {e}", exc_info=True)

            # 5. Text Processing Orchestrator
            self.text_processor = SD3TextProcessingOrchestrator(
                clip_l_model=clip.cond_stage_model.clip_l, tokenizer_l=clip.tokenizer.clip_l,
                clip_g_model=clip.cond_stage_model.clip_g, tokenizer_g=clip.tokenizer.clip_g,
                t5_model=clip.cond_stage_model.t5xxl, tokenizer_t5=clip.tokenizer.t5xxl,
                config=self.config
            )

            # 6. Forge Objects Container
            self.forge_objects = SD3ForgeObjects(
                 transformer=unet_patcher, vae=vae, clip=clip,
                 clip_l=clip_l, clip_g=clip_g, t5xxl=t5xxl,
                 tokenizer_l=tokenizer_l, tokenizer_g=tokenizer_g, tokenizer_t5=tokenizer_t5
            )

            # Store copies for LoRA management / state restoration
            self.forge_objects_original = self.forge_objects.shallow_copy()
            self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

            # --- Legacy WebUI Compatibility ---
            self.is_sd3 = True # Flag for potential use in WebUI logic

        except KeyError as e:
            logger.error(f"Missing expected HuggingFace component key: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize {self.engine_name} due to missing component: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during {self.engine_name} initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize {self.engine_name}: {e}") from e

        logger.info(f"{self.engine_name} initialized successfully.")

    def set_clip_skip(self, clip_skip: int):
        """Applies CLIP skip to the text processing engines."""
        if hasattr(self, 'text_processor'):
            self.text_processor.set_clip_skip(clip_skip)
        else:
            logger.warning("Text processor not initialized, cannot set CLIP skip.")

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Generates the conditioning tensors (cross-attention inputs and pooled vectors)
        for the given prompt(s).
        """
        if not hasattr(self, 'text_processor'):
            raise RuntimeError("Text processor not initialized.")

        logger.debug(f"Generating learned conditioning for prompt: {' | '.join(prompt) if isinstance(prompt, list) else prompt}")
        # Delegate processing to the orchestrator
        # The orchestrator handles model loading to GPU internally
        conditioning = self.text_processor.process_prompt(prompt)
        logger.debug(f"Conditioning generated - Keys: {list(conditioning.keys())}, "
                     f"CrossAttn shape: {conditioning.get('crossattn', torch.zeros(0)).shape}, "
                     f"Vector shape: {conditioning.get('vector', torch.zeros(0)).shape}")
        return conditioning

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> Tuple[int, int]:
        """
        Returns the token count and maximum allowed tokens for UI display,
        primarily based on the T5 tokenizer.
        """
        if not hasattr(self, 'text_processor'):
             logger.warning("Text processor not initialized, returning default UI lengths.")
             return 0, 255 # Fallback
        return self.text_processor.get_prompt_lengths_on_ui(prompt)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an image tensor (B, C, H, W) into latents."""
        if not self.forge_objects or not self.forge_objects.vae:
             raise RuntimeError("VAE not initialized in ForgeObjects.")

        vae = self.forge_objects.vae
        # Expected input for VAE encode: (B, H, W, C) in range [0, 1]
        # Input x is usually in range [-1, 1] with shape (B, C, H, W)
        logger.debug(f"Encoding image batch with shape: {x.shape}")
        x = x.movedim(1, -1) # (B, H, W, C)
        x = x * 0.5 + 0.5 # Range [0, 1]
        memory_management.load_model_gpu(vae) # Ensure VAE is loaded
        sample = vae.encode(x) # VAE Patcher handles internal model call

        # Process output if needed (depends on VAE patcher implementation)
        # The original code had a process_in step, check if vae.encode handles this
        if hasattr(vae.first_stage_model, 'process_in'):
             logger.debug("Applying VAE process_in.")
             sample = vae.first_stage_model.process_in(sample)

        logger.debug(f"Encoded latent shape: {sample.shape}")
        return sample.to(x.device) # Ensure device consistency

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes a latent tensor (B, Z, H/f, W/f) into an image tensor."""
        if not self.forge_objects or not self.forge_objects.vae:
             raise RuntimeError("VAE not initialized in ForgeObjects.")

        vae = self.forge_objects.vae
        logger.debug(f"Decoding latent batch with shape: {x.shape}")
        memory_management.load_model_gpu(vae) # Ensure VAE is loaded

        # Process input if needed (depends on VAE patcher implementation)
        # The original code had a process_out step
        if hasattr(vae.first_stage_model, 'process_out'):
             logger.debug("Applying VAE process_out.")
             x = vae.first_stage_model.process_out(x)

        sample = vae.decode(x) # VAE Patcher handles internal model call

        # Output should be in range [-1, 1] with shape (B, C, H, W)
        # VAE Patcher decode usually outputs (B, H, W, C) in [0, 1]
        sample = sample.movedim(-1, 1) # (B, C, H, W)
        sample = sample * 2.0 - 1.0 # Range [-1, 1]

        logger.debug(f"Decoded image shape: {sample.shape}")
        return sample.to(x.device) # Ensure device consistency

    def load_lora(self, lora_name: str, weight: float):
        """Placeholder for applying a LoRA/adapter."""
        # TODO: Implement actual LoRA loading using backend patcher mechanisms
        logger.info(f"Applying LoRA '{lora_name}' with weight {weight} (Placeholder)")
        # This should interact with self.forge_objects.transformer, self.forge_objects.clip etc.
        # and potentially store state in self.forge_objects_after_applying_lora
        self.applied_loras[lora_name] = weight
        # In a real implementation, you'd patch the models here.
        # self.forge_objects = self.forge_objects_after_applying_lora.shallow_copy() # Update active objects

    def unload_loras(self):
        """Placeholder for removing all applied LoRAs/adapters."""
        # TODO: Implement actual LoRA unloading
        logger.info("Unloading all LoRAs (Placeholder)")
        self.applied_loras = {}
        # Restore original state
        # self.forge_objects = self.forge_objects_original.shallow_copy() # Restore objects
        # self.forge_objects_after_applying_lora = self.forge_objects_original.shallow_copy() # Reset LoRA state copy

# --- Initialization Hook ---
# Ensure the patching happens when the module is loaded or via an explicit setup call
StableDiffusion3Enhanced.setup_huggingface_heuristic_patches()
