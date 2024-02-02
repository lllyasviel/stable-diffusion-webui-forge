from dataclasses import dataclass


@dataclass
class ControlNetUnitGradioState:
    enabled: bool = True
    preprocessor_name: str = "None"
    control_model_name: str = "None"
    weight: float = 1.0
    image = None
    mask = None
    resize_mode: str = "Crop and Resize"
    processor_resolution: int = -1
    slider_1: float = -1
    slider_2: float = -1
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    pixel_perfect: bool = False
    control_mode: str = "Balanced"
