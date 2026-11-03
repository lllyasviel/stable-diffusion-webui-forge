
def gradio_compile(items, prefix):
    names = []
    for k, v in items["required"].items():
        t = v[0]
        d = v[1] if len(v) > 1 else None
        if prefix != '':
            name = (prefix + '_' + k).replace(' ', '_').lower()
        else:
            name = k.replace(' ', '_').lower()

        title = name.replace('_', ' ').title()

        if t == 'INT':
            default = int(d['default'])
            min = int(d['min'])
            max = int(d['max'])
            step = int(d.get('step', 1))
            print(f'{name} = gr.Slider(label=\'{title}\', minimum={min}, maximum={max}, step={step}, value={default})')
            names.append(name)
        elif t == 'FLOAT':
            default = float(d['default'])
            min = float(d['min'])
            max = float(d['max'])
            step = float(d.get('step', 0.001))
            print(f'{name} = gr.Slider(label=\'{title}\', minimum={min}, maximum={max}, step={step}, value={default})')
            names.append(name)
        elif isinstance(t, list):
            print(f'{name} = gr.Radio(label=\'{title}\', choices={str(t)}, value=\'{t[0]}\')')
            names.append(name)
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

    return ['enabled'] + names


def print_info_text(name_list, prefix):
    print(', '.join(name_list))
    print('p.extra_generation_params.update(dict(')
    for n in name_list:
        print(prefix + '_' + n + ' = ' + n + ', ')
    print(')')
    return


# from modules_forge.gradio_compile import gradio_compile
# ps = []
# ps += gradio_compile(SVD_img2vid_Conditioning.INPUT_TYPES(), prefix='')
# ps += gradio_compile(KSampler.INPUT_TYPES(), prefix='sampling')
# ps += gradio_compile(VideoLinearCFGGuidance.INPUT_TYPES(), prefix='guidance')
# print(', '.join(ps))
# print_info_text(ps, '123')
