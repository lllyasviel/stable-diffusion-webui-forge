import os
import time
import gradio as gr
import importlib.util

from threading import Thread, Event


spaces = []


def build_html(title, url=None):
    if isinstance(url, str):
        return f'<div>{title}</div><div>Currently Running: <a href="{url}" style="color: green;">{url}</a></div>'
    else:
        return f'<div>{title}</div><div style="color: grey;">Currently Offline</div>'


class ForgeSpace:
    def __init__(self, root_path, title, **kwargs):
        self.title = title
        self.root_path = root_path
        self.is_running = False
        self.gradio_metas = None

        self.label = gr.HTML(build_html(title=title, url=None), elem_classes=['forge_space_label'])
        self.btn_install = gr.Button('Install', elem_classes=['forge_space_btn'])
        self.btn_uninstall = gr.Button('Uninstall', elem_classes=['forge_space_btn'])
        self.btn_launch = gr.Button('Launch', elem_classes=['forge_space_btn'])
        self.btn_terminate = gr.Button('Terminate', elem_classes=['forge_space_btn'])

        self.btn_launch.click(self.run, inputs=[], outputs=[self.label])
        self.btn_terminate.click(self.terminate, inputs=[], outputs=[self.label])

        return

    def terminate(self):
        self.is_running = False
        while self.gradio_metas is not None:
            time.sleep(0.1)
        html = build_html(title=self.title, url=None)
        return html

    def run(self):
        self.is_running = True
        Thread(target=self.gradio_worker).start()
        while self.gradio_metas is None:
            time.sleep(0.1)

        html = build_html(title=self.title, url=self.gradio_metas[1])
        return html

    def gradio_worker(self):
        file_path = os.path.join(self.root_path, 'app.py')
        module_name = 'app'
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        demo = getattr(module, 'demo')

        self.gradio_metas = demo.launch(inbrowser=True, prevent_thread_lock=True)

        while self.is_running:
            time.sleep(0.1)

        demo.close()
        self.gradio_metas = None
        return


# Event to signal the thread to stop
stop_event = Event()


def open_another():
    # Define the path to the app.py file
    file_path = os.path.join('extensions-builtin', 'forge_space_test', 'app.py')

    # Specify the module name (this can be anything you want)
    module_name = 'app'

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    demo = getattr(module, 'demo')

    metas = demo.launch(inbrowser=True, prevent_thread_lock=True)

    # Loop to keep the UI open
    while not stop_event.is_set():
        stop_event.wait(0.1)  # Checks every 100ms if the event is set

    demo.close()
    print('ended')


def main_entry():
    global spaces

    from modules.extensions import extensions

    tagged_extensions = {}

    for ex in extensions:
        if ex.enabled and ex.is_forge_space:
            tag = ex.space_meta['tag']

            if tag not in tagged_extensions:
                tagged_extensions[tag] = []

            tagged_extensions[tag].append(ex)

    for tag, exs in tagged_extensions.items():
        with gr.Accordion(tag):
            for ex in exs:
                with gr.Row(equal_height=True):
                    space = ForgeSpace(root_path=ex.path, **ex.space_meta)
                    spaces.append(space)

    # space = ForgeSpace(root_path=ex.path, **ex.space_meta)

    # btn = gr.Button('Run')
    # thread = Thread(target=open_another)
    # btn.click(thread.start)
    #
    # btn2 = gr.Button('Close')
    # btn2.click(fn=stop_event.set)  # Signal the thread to stop
