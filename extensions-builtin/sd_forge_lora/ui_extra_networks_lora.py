import os

import network
import networks

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js
from ui_edit_user_metadata import LoraUserMetadataEditor


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')
        self.allow_negative_prompt = True

    def refresh(self):
        networks.list_available_networks()

    def create_item(self, name, index=None, enable_filter=True):
        lora_on_disk = networks.available_networks.get(name)
        if lora_on_disk is None:
            return

        path, ext = os.path.splitext(lora_on_disk.filename)

        alias = lora_on_disk.get_alias()

        search_terms = [self.search_terms_from_path(lora_on_disk.filename)]
        if lora_on_disk.hash:
            search_terms.append(lora_on_disk.hash)
        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "shorthash": lora_on_disk.shorthash,
            "preview": self.find_preview(path) or self.find_embedded_preview(path, name, lora_on_disk.metadata),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        item["prompt"] = quote_js(f"<lora:{alias}:") + " + " + (str(preferred_weight) if preferred_weight else "opts.extra_networks_default_multiplier") + " + " + quote_js(">")

        if activation_text:
            item["prompt"] += " + " + quote_js(" " + activation_text)

        negative_prompt = item["user_metadata"].get("negative text", "")
        item["negative_prompt"] = quote_js(negative_prompt)

        #   filter displayed loras by UI setting
        sd_version = item["user_metadata"].get("sd version")
        if sd_version in network.SdVersion.__members__:
            item["sd_version"] = sd_version
            sd_version = network.SdVersion[sd_version]
        else:
            sd_version = lora_on_disk.sd_version        #   use heuristics
            #sd_version = network.SdVersion.Unknown     #   avoid heuristics 

        item["sd_version_str"] = str(sd_version)

        return item

    def list_items(self):
        # instantiate a list to protect against concurrent modification
        names = list(networks.available_networks)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)
