import sys

import modules.upscaler
from modules import devices, errors, script_callbacks, shared, upscaler_utils


class UpscalerScuNET(modules.upscaler.Upscaler):
    def __init__(self, dirname):
        self.name = "ScuNET"
        self.model_name = "ScuNET GAN"
        self.model_name2 = "ScuNET PSNR"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
        self.model_url2 = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
        self.user_path = dirname
        super().__init__()
        self._scalers_initialized = False
        self.scalers = []

    def _init_scalers(self):
        if self._scalers_initialized:
            return
        # Defer modelloader import in case it's heavy
        from modules import modelloader
        model_paths = self.find_models(ext_filter=[".pth"])
        scalers = []
        add_model2 = True
        for file in model_paths:
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)
            if name == self.model_name2 or file == self.model_url2:
                add_model2 = False
            try:
                scaler_data = modules.upscaler.UpscalerData(name, file, self, 4)
                scalers.append(scaler_data)
            except Exception:
                errors.report(f"Error loading ScuNET model: {file}", exc_info=True)
        if add_model2:
            scaler_data2 = modules.upscaler.UpscalerData(self.model_name2, self.model_url2, self)
            scalers.append(scaler_data2)
        self.scalers = scalers
        self._scalers_initialized = True

    def get_scalers(self):
        self._init_scalers()
        return self.scalers

    def do_upscale(self, img, selected_file):
        import PIL.Image
        self._init_scalers()
        devices.torch_gc()
        try:
            model = self._load_scunet_model(selected_file)
        except Exception as e:
            print(f"ScuNET: Unable to load model from {selected_file}: {e}", file=sys.stderr)
            return img

        tile_size = shared.opts.SCUNET_tile if shared.opts.SCUNET_tile is not None else 256
        tile_overlap = shared.opts.SCUNET_tile_overlap if shared.opts.SCUNET_tile_overlap is not None else 8

        img = upscaler_utils.upscale_2(
            img,
            model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=1,  # ScuNET is a denoising model, not an upscaler
            desc='ScuNET',
        )
        devices.torch_gc()
        return img

    def _load_scunet_model(self, path: str):
        from modules import modelloader
        device = devices.get_device_for('scunet')
        # Always use a valid string for model_dir
        model_dir = getattr(self, 'model_download_path', None)
        if not isinstance(model_dir, str) or not model_dir:
            model_dir = self.user_path if isinstance(self.user_path, str) else "."
        if path.startswith("http"):
            filename = modelloader.load_file_from_url(self.model_url, model_dir=model_dir, file_name=f"{self.name}.pth")
        else:
            filename = path
        return modelloader.load_spandrel_model(filename, device=device, expected_architecture='SCUNet')


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("SCUNET_tile", shared.OptionInfo(256, "Tile size for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")).info("0 = no tiling"))
    shared.opts.add_option("SCUNET_tile_overlap", shared.OptionInfo(8, "Tile overlap for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=('upscaling', "Upscaling")).info("Low values = visible seam"))


script_callbacks.on_ui_settings(on_ui_settings)
