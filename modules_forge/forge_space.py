import gradio as gr
from threading import Thread, Event

# Event to signal the thread to stop
stop_event = Event()


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
