import os
import enum

from modules import sd_models, cache, errors, hashes, shared


metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

class SdVersion(enum.Enum):
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SDXL = 4
#    SD3 = 5
    Flux = 6

class NetworkOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        def read_metadata():
            metadata = sd_models.read_metadata_from_safetensors(filename)

            return metadata

        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.alias = self.metadata.get('ss_output_name', self.name)

        self.hash = None
        self.shorthash = None
        self.set_hash(
            self.metadata.get('sshs_model_hash') or
            hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or
            ''
        )

        self.sd_version = self.detect_version()
    
    def detect_version(self):
        if str(self.metadata.get('modelspec.implementation', '')) == 'https://github.com/black-forest-labs/flux':
            return SdVersion.Flux
        elif str(self.metadata.get('modelspec.architecture', '')) == 'flux-1-dev/lora':
            return SdVersion.Flux
        elif str(self.metadata.get('modelspec.architecture', '')) == 'stable-diffusion-xl-v1-base/lora':
            return SdVersion.SDXL
        elif str(self.metadata.get('ss_base_model_version', '')).startswith('sdxl_'):
            return SdVersion.SDXL
        elif str(self.metadata.get('ss_v2', '')) == 'True':
            return SdVersion.SD2
        elif str(self.metadata.get('modelspec.architecture', '')) == 'stable-diffusion-v1/lora':
            return SdVersion.SD1

        return SdVersion.Unknown
    
    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            import networks
            networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    def get_alias(self):
        import networks
        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in networks.forbidden_network_aliases:
            return self.name
        else:
            return self.alias


class Network:
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.bundle_embeddings = {}
        self.mtime = None
        self.mentioned_name = None
