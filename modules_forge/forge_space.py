import gradio as gr
from threading import Thread


def open_another():
    def greet(name):
        return "Hello " + name + "!"

    demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    demo.launch(inbrowser=True)


def main_ui():
    btn = gr.Button('Hello')
    btn.click(lambda: Thread(open_another).start())
    return
