import os
import modules.textual_inversion.textual_inversion

from modules.shared import cmd_opts
from modules import ui_extra_networks, shared
from modules.ui_extra_networks import quote_js


embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()
embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)
embedding_db.load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self):
        embedding_db.load_textual_inversion_embeddings(force_reload=True, sync_with_sd_model=False)

    def create_item(self, name, index=None, enable_filter=True):
        embedding = embedding_db.word_embeddings.get(name)
        if embedding is None:
            return

        path, ext = os.path.splitext(embedding.filename)
        search_terms = [self.search_terms_from_path(embedding.filename)]
        if embedding.hash:
            search_terms.append(embedding.hash)
        return {
            "name": name,
            "filename": embedding.filename,
            "shorthash": embedding.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "prompt": quote_js(embedding.name),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(embedding.filename)},
        }

    def list_items(self):
        # instantiate a list to protect against concurrent modification
        names = list(embedding_db.word_embeddings)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return list(embedding_db.embedding_dirs)
