
def gradio_compile(items, prefix):
    for k, v in items["required"].items():
        if len(v) == 2:
            t, d = v
            if t == 'INT':
                name = (prefix + '_' + k).replace(' ', '_').lower()
                title = name.replace('_', ' ').capitalize()
                default = int(d['default'])
                min = int(d['min'])
                max = int(d['max'])
                step = int(d.get('step', 1))
                print(f'{name} = gr.Slider(label=\'{title}\', minimum={min}, maximum={max}, step={step}, value={default})')
    return
