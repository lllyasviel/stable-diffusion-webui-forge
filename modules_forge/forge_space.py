import os
import gradio as gr
import importlib.util

from modules import extensions as a1111_extensions
from threading import Thread, Event

a = 0

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


def main_ui():
    btn = gr.Button('Run')
    thread = Thread(target=open_another)
    btn.click(thread.start)

    btn2 = gr.Button('Close')
    btn2.click(fn=stop_event.set)  # Signal the thread to stop
