
def gradio_compile(items, prefix):
    for k, v in items["required"].items():
        t = v[0]
        d = v[1] if len(v) > 1 else None
        if prefix != '':
            name = (prefix + k).replace(' ', '_').lower()
        else:
            name = k.replace(' ', '_').lower()
        title = name.replace('_', ' ').title()

        if t == 'INT':
            default = int(d['default'])
            min = int(d['min'])
            max = int(d['max'])
            step = int(d.get('step', 1))
            print(f'{name} = gr.Slider(label=\'{title}\', minimum={min}, maximum={max}, step={step}, value={default})')
        elif t == 'FLOAT':
            default = float(d['default'])
            min = float(d['min'])
            max = float(d['max'])
            step = float(d.get('step', 0.001))
            print(f'{name} = gr.Slider(label=\'{title}\', minimum={min}, maximum={max}, step={step}, value={default})')
        elif isinstance(t, list):
            print(f'{name} = gr.Radio(label=\'{title}\', choices={str(t)}, value=\'{t[0]}\')')
        elif t == 'MODEL':
            pass
        elif t == 'CONDITIONING':
            pass
        elif t == 'LATENT':
            pass
        elif t == 'CLIP_VISION':
            pass
        elif t == 'IMAGE':
            pass
        elif t == 'VAE':
            pass
        else:
            print('error ' + str(t))

    return
