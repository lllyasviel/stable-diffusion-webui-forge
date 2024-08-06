import gradio as gr


def make_checkpoint_manager_ui():
    gr.Radio(label='Model Type', choices=['StableDiffusion 1.5', 'StableDiffusion XL'], value='StableDiffusion 1.5')
    return


def forge_main_entry():
    print('Hello World!')
    return
