import gradio as gr
from threading import Thread


def open_another():
    def update(name):
        return f"Welcome to Gradio, {name}!"

    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Run** to see the output.")
        with gr.Row():
            inp = gr.Textbox(placeholder="What is your name?")
            out = gr.Textbox()
        btn = gr.Button("Run")
        btn.click(fn=update, inputs=inp, outputs=out)

    demo.launch(inbrowser=True)


def main_ui():
    btn = gr.Button('Hello')
    thread = Thread(target=open_another)
    btn.click(thread.start)
    return
