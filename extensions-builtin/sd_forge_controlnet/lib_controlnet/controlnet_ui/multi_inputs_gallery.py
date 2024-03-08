import gradio as gr
from typing import Optional


class MultiInputsGallery:
    """A gallery object that accepts multiple input images."""

    def __init__(self, row: int = 2, column: int = 4, **group_kwargs) -> None:
        self.gallery_row_num = row
        self.gallery_column_num = column
        self.group_kwargs = group_kwargs

        self.group = None
        self.input_gallery = None
        self.upload_button = None
        self.clear_button = None
        self.render()

    def render(self):
        with gr.Group(**self.group_kwargs) as self.group:
            with gr.Column():
                self.input_gallery = gr.Gallery(
                    columns=[self.gallery_column_num],
                    rows=[self.gallery_row_num],
                    object_fit="contain",
                    height="auto",
                    label="Images",
                )
                with gr.Row():
                    self.upload_button = gr.UploadButton(
                        "Upload Images",
                        file_types=["image"],
                        file_count="multiple",
                    )
                    self.clear_button = gr.Button("Clear Images")

    def register_callbacks(self, change_trigger: Optional[dict] = None):
        """Register callbacks on multiple images upload.
        Argument:
            - change_trigger: An optional gradio callback param dict to be called
            after gallery content change. This is necessary as gallery has no
            event subscriber. If the state change of gallery needs to be observed,
            the caller needs to pass a change trigger to observe the change.
        """
        handle1 = self.clear_button.click(
            fn=lambda: [],
            inputs=[],
            outputs=[self.input_gallery],
        )

        def upload_file(files, current_files):
            return {file_d["name"] for file_d in current_files} | {
                file.name for file in files
            }

        handle2 = self.upload_button.upload(
            upload_file,
            inputs=[self.upload_button, self.input_gallery],
            outputs=[self.input_gallery],
            queue=False,
        )

        if change_trigger:
            for handle in (handle1, handle2):
                handle.then(**change_trigger)
