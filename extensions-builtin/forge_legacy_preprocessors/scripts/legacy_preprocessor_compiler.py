# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format manually
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

import json
from legacy_preprocessors.preprocessor_meta import ui_preprocessor_keys, reverse_preprocessor_aliases, preprocessor_aliases
from legacy_preprocessors.preprocessor import model_free_preprocessors, no_control_mode_preprocessors, preprocessor_sliders_config, preprocessor_filters, preprocessor_filters_aliases


cn_preprocessor_modules = '''
    "none": lambda x, *args, **kwargs: (x, True),
    "canny": canny,
    "depth": midas,
    "depth_leres": functools.partial(leres, boost=False),
    "depth_leres++": functools.partial(leres, boost=True),
    "depth_hand_refiner": g_hand_refiner_model.run_model,
    "depth_anything": functools.partial(depth_anything, colored=False),
    "hed": hed,
    "hed_safe": hed_safe,
    "mediapipe_face": mediapipe_face,
    "mlsd": mlsd,
    "normal_map": midas_normal,
    "openpose": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=False),
    "openpose_hand": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=False),
    "openpose_face": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=True),
    "openpose_faceonly": functools.partial(g_openpose_model.run_model, include_body=False, include_hand=False, include_face=True),
    "openpose_full": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=True),
    "dw_openpose_full": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=True, use_dw_pose=True),
    "animal_openpose": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=False, use_animal_pose=True),
    "clip_vision": functools.partial(clip, config='clip_vitl'),
    "revision_clipvision": functools.partial(clip, config='clip_g'),
    "revision_ignore_prompt": functools.partial(clip, config='clip_g'),
    "ip-adapter_clip_sd15": functools.partial(clip, config='clip_h'),
    "ip-adapter_clip_sdxl_plus_vith": functools.partial(clip, config='clip_h'),
    "ip-adapter_clip_sdxl": functools.partial(clip, config='clip_g'),
    "ip-adapter_face_id": g_insight_face_model.run_model,
    "ip-adapter_face_id_plus": face_id_plus,
    "instant_id_face_keypoints": functools.partial(g_insight_face_instant_id_model.run_model_instant_id, return_keypoints=True),
    "instant_id_face_embedding": functools.partial(g_insight_face_instant_id_model.run_model_instant_id, return_keypoints=False),
    "color": color,
    "pidinet": pidinet,
    "pidinet_safe": pidinet_safe,
    "pidinet_sketch": pidinet_ts,
    "pidinet_scribble": scribble_pidinet,
    "scribble_xdog": scribble_xdog,
    "scribble_hed": scribble_hed,
    "segmentation": uniformer,
    "threshold": threshold,
    "depth_zoe": zoe_depth,
    "normal_bae": normal_bae,
    "oneformer_coco": oneformer_coco,
    "oneformer_ade20k": oneformer_ade20k,
    "lineart": lineart,
    "lineart_coarse": lineart_coarse,
    "lineart_anime": lineart_anime,
    "lineart_standard": lineart_standard,
    "shuffle": shuffle,
    "tile_resample": tile_resample,
    "invert": invert,
    "lineart_anime_denoise": lineart_anime_denoise,
    "reference_only": identity,
    "reference_adain": identity,
    "reference_adain+attn": identity,
    "inpaint": identity,
    "inpaint_only": identity,
    "inpaint_only+lama": lama_inpaint,
    "tile_colorfix": identity,
    "tile_colorfix+sharp": identity,
    "recolor_luminance": recolor_luminance,
    "recolor_intensity": recolor_intensity,
    "blur_gaussian": blur_gaussian,
    "anime_face_segment": anime_face_segment,
    "densepose": functools.partial(densepose, cmap="viridis"),
    "densepose_parula": functools.partial(densepose, cmap="parula"),
    "te_hed":te_hed,
'''

cn_preprocessor_unloadable = '''
    "hed": unload_hed,
    "fake_scribble": unload_hed,
    "mlsd": unload_mlsd,
    "clip_vision": functools.partial(unload_clip, config='clip_vitl'),
    "revision_clipvision": functools.partial(unload_clip, config='clip_g'),
    "revision_ignore_prompt": functools.partial(unload_clip, config='clip_g'),
    "ip-adapter_clip_sd15": functools.partial(unload_clip, config='clip_h'),
    "ip-adapter_clip_sdxl_plus_vith": functools.partial(unload_clip, config='clip_h'),
    "ip-adapter_face_id_plus": functools.partial(unload_clip, config='clip_h'),
    "ip-adapter_clip_sdxl": functools.partial(unload_clip, config='clip_g'),
    "depth": unload_midas,
    "depth_leres": unload_leres,
    "depth_anything": unload_depth_anything,
    "normal_map": unload_midas,
    "pidinet": unload_pidinet,
    "openpose": g_openpose_model.unload,
    "openpose_hand": g_openpose_model.unload,
    "openpose_face": g_openpose_model.unload,
    "openpose_full": g_openpose_model.unload,
    "dw_openpose_full": g_openpose_model.unload,
    "animal_openpose": g_openpose_model.unload,
    "segmentation": unload_uniformer,
    "depth_zoe": unload_zoe_depth,
    "normal_bae": unload_normal_bae,
    "oneformer_coco": unload_oneformer_coco,
    "oneformer_ade20k": unload_oneformer_ade20k,
    "lineart": unload_lineart,
    "lineart_coarse": unload_lineart_coarse,
    "lineart_anime": unload_lineart_anime,
    "lineart_anime_denoise": unload_lineart_anime_denoise,
    "inpaint_only+lama": unload_lama_inpaint,
    "anime_face_segment": unload_anime_face_segment,
    "densepose": unload_densepose,
    "densepose_parula": unload_densepose,
    "depth_hand_refiner": g_hand_refiner_model.unload,
    "te_hed":unload_te_hed,
'''


def compile_first_round(text):
    lines = text.splitlines()
    dd = {}
    for l in lines:
        sp = l.split('":')
        if len(sp) == 2:
            k, v = sp
            k = k.strip(' ",')
            v = v.strip(' ",')
            dd[k] = v
    return dd


cn_preprocessor_modules = compile_first_round(cn_preprocessor_modules)
cn_preprocessor_unloadable = compile_first_round(cn_preprocessor_unloadable)


def special_get(d, k, default=None):
    k1 = k
    k2 = preprocessor_aliases.get(k, k)
    k3 = reverse_preprocessor_aliases.get(k, k)

    for pk in [k1, k2, k3]:
        if pk in d:
            return d[pk]

    return default


def special_judge_in(d, k):
    k1 = k
    k2 = preprocessor_aliases.get(k, k)
    k3 = reverse_preprocessor_aliases.get(k, k)

    for pk in [k1, k2, k3]:
        if pk in d:
            return True

    return False


legacy_preprocessors = {}

for name in ui_preprocessor_keys:
    call_function = special_get(cn_preprocessor_modules, name, None)
    assert call_function is not None
    unload_function = special_get(cn_preprocessor_unloadable, name, 'None')

    model_free = special_judge_in(model_free_preprocessors, name)
    no_control_mode = special_judge_in(no_control_mode_preprocessors, name)
    slider_config = special_get(preprocessor_sliders_config, name, [])

    resolution = slider_config[0] if len(slider_config) > 0 else None
    slider_1 = slider_config[1] if len(slider_config) > 1 else None
    slider_2 = slider_config[2] if len(slider_config) > 2 else None
    slider_3 = slider_config[3] if len(slider_config) > 3 else None

    legacy_preprocessors[name] = dict(
        name=name,
        call_function='***' + call_function + '***',
        unload_function='***' + unload_function + '***',
        managed_model='***None***',
        model_free=model_free,
        no_control_mode=no_control_mode,
        resolution=resolution,
        slider_1=slider_1,
        slider_2=slider_2,
        slider_3=slider_3,
        priority=0,
        tag=None
    )


for tag, best in preprocessor_filters.items():
    bp = special_get(legacy_preprocessors, best, None)
    if bp is not None:
        bp['priority'] = 100

for tag, best in preprocessor_filters.items():
    marks = [tag.lower()] + preprocessor_filters_aliases.get(tag.lower(), [])
    for k, p in legacy_preprocessors.items():
        if any(x.lower() in k.lower() for x in marks):
            p['tag'] = tag


compiled_filename = __file__.replace('compiler', 'compiled')

with open(compiled_filename, 'wt') as fp:
    result = json.dumps(legacy_preprocessors, indent=4).replace('null', 'None')\
        .replace('false', 'False').replace('true', 'True').replace('***"', '').replace('"***', '').replace('\\"', '"').replace('"Balanced"', 'Balanced')
    fp.write('import functools\nfrom legacy_preprocessors.preprocessor import *\n\n\nlegacy_preprocessors = ' + result)

print('ok')
