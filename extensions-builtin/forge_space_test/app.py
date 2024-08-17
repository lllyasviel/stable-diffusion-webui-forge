import gradio as gr


def update(name):
    return f"Welcome to Gradio, {name}!"


with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


if __name__ == "__main__":
    metas = demo.launch(inbrowser=True)
