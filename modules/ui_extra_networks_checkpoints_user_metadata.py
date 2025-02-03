import gradio as gr

from modules import ui_extra_networks_user_metadata, sd_vae, shared
from modules.ui_components import ToolButton
from modules_forge import main_entry

refresh_symbol = '\U0001f504'  # 🔄

class CheckpointUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui, tabname, page):
        super().__init__(ui, tabname, page)

        self.select_vae = None
        self.sd_version = 'Unknown'

    def save_user_metadata(self, name, desc, notes, vae, sd_version):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["notes"] = notes
        user_metadata["vae_te"] = vae
        user_metadata["sd_version_str"] = 'SdVersion.' + sd_version

        self.write_user_metadata(name, user_metadata)

    def put_values_into_components(self, name):
        user_metadata = self.get_user_metadata(name)
        values = super().put_values_into_components(name)
        
        vae = user_metadata.get('vae_te', None)
        if vae is None:     # fallback to old type
            vae = user_metadata.get('vae', None)
            if vae is not None:
                if isinstance(vae, str):
                    vae = [vae]

        version = user_metadata.get('sd_version_str', '')
        if version == '':
            version = 'Unknown'
        else:
            version = version.replace('SdVersion.', '')

        return [
            *values[0:5],
            vae,
            version,
        ]

    def create_editor(self):    #happens before main_entry.modules_list is filled
        modules_list = ['Built in']
        if main_entry.module_list == {}:
            _, modules = main_entry.refresh_models()
            modules_list += list(modules)
        else:
            modules_list += list(main_entry.module_list.keys())
        
        def refreshModules ():
            return gr.update(choices=['Built in'] + list(main_entry.module_list.keys()))

        self.create_default_editor_elems()

        self.sd_version = gr.Radio(['SD1', 'SD2', 'SDXL', 'Flux', 'Unknown'], value='Unknown', label='Base model', interactive=True)

        with gr.Row():
            self.select_vae = gr.Dropdown(choices=modules_list, value=None, label="Preferred VAE / Text encoder(s)", elem_id="checpoint_edit_user_metadata_preferred_vae", multiselect=True)
            self.refresh = ToolButton(refresh_symbol)
            
            self.refresh.click(fn=refreshModules, outputs=self.select_vae, show_progress='hidden')

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        self.create_default_buttons()

        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.select_vae,
            self.sd_version,
        ]

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=viewed_components)\
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        edited_components = [
            self.edit_description,
            self.edit_notes,
            self.select_vae,
            self.sd_version,
        ]

        self.setup_save_handler(self.button_save, self.save_user_metadata, edited_components)
