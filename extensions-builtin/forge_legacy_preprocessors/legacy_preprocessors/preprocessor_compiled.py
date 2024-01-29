from legacy_preprocessors.preprocessor import *


legacy_preprocessors = {
    # "none": {
    #     "name": "none",
    #     "call_function": lambda x, *args, **kwargs: (x, True),
    #     "unload_function": None,
    #     "managed_model": None,
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": []
    # },
    "invert (from white bg & black line)": {
        "name": "invert (from white bg & black line)",
        "call_function": invert,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Canny", "Lineart", "Scribble", "Sketch", "MLSD",
        ]
    },
    "animal_openpose": {
        "name": "animal_openpose",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=False, use_animal_pose=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "blur_gaussian": {
        "name": "blur_gaussian",
        "call_function": blur_gaussian,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": {
            "name": "Sigma",
            "min": 0.01,
            "max": 64.0,
            "value": 9.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Tile", "Blur",
        ]
    },
    # "canny": {
    #     "name": "canny",
    #     "call_function": canny,
    #     "unload_function": None,
    #     "managed_model": "model_canny",
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": {
    #         "name": "Preprocessor Resolution",
    #         "value": 512,
    #         "min": 64,
    #         "max": 2048
    #     },
    #     "slider_1": {
    #         "name": "Canny Low Threshold",
    #         "value": 100,
    #         "min": 1,
    #         "max": 255
    #     },
    #     "slider_2": {
    #         "name": "Canny High Threshold",
    #         "value": 200,
    #         "min": 1,
    #         "max": 255
    #     },
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": [
    #         "Canny"
    #     ]
    # },
    "densepose (pruple bg & purple torso)": {
        "name": "densepose (pruple bg & purple torso)",
        "call_function": functools.partial(densepose, cmap="viridis"),
        "unload_function": unload_densepose,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "densepose_parula (black bg & blue torso)": {
        "name": "densepose_parula (black bg & blue torso)",
        "call_function": functools.partial(densepose, cmap="parula"),
        "unload_function": unload_densepose,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "depth_anything": {
        "name": "depth_anything",
        "call_function": functools.partial(depth_anything, colored=False),
        "unload_function": unload_depth_anything,
        "managed_model": "model_depth_anything",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_hand_refiner": {
        "name": "depth_hand_refiner",
        "call_function": g_hand_refiner_model.run_model,
        "unload_function": g_hand_refiner_model.unload,
        "managed_model": "g_hand_refiner_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_leres": {
        "name": "depth_leres",
        "call_function": functools.partial(leres, boost=False),
        "unload_function": unload_leres,
        "managed_model": "model_leres",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_2": {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_leres++": {
        "name": "depth_leres++",
        "call_function": functools.partial(leres, boost=True),
        "unload_function": None,
        "managed_model": "model_leres",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": {
            "name": "Remove Near %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_2": {
            "name": "Remove Background %",
            "min": 0,
            "max": 100,
            "value": 0,
            "step": 0.1
        },
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "depth_midas": {
        "name": "depth_midas",
        "call_function": midas,
        "unload_function": unload_midas,
        "managed_model": "model_midas",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Depth"
        ]
    },
    "depth_zoe": {
        "name": "depth_zoe",
        "call_function": zoe_depth,
        "unload_function": unload_zoe_depth,
        "managed_model": "model_zoe_depth",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Depth"
        ]
    },
    "dw_openpose_full": {
        "name": "dw_openpose_full",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=True, use_dw_pose=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": 'g_openpose_model',
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "inpaint_global_harmonious": {
        "name": "inpaint_global_harmonious",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Inpaint"
        ]
    },
    "inpaint_only": {
        "name": "inpaint_only",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Inpaint"
        ]
    },
    "inpaint_only+lama": {
        "name": "inpaint_only+lama",
        "call_function": lama_inpaint,
        "unload_function": unload_lama_inpaint,
        "managed_model": "model_lama",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Inpaint"
        ]
    },
    "instant_id_face_embedding": {
        "name": "instant_id_face_embedding",
        "call_function": functools.partial(g_insight_face_instant_id_model.run_model_instant_id, return_keypoints=False),
        "unload_function": None,
        "managed_model": "g_insight_face_instant_id_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Instant_ID"
        ]
    },
    "instant_id_face_keypoints": {
        "name": "instant_id_face_keypoints",
        "call_function": functools.partial(g_insight_face_instant_id_model.run_model_instant_id, return_keypoints=True),
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Instant_ID"
        ]
    },
    "ip-adapter_clip_sd15": {
        "name": "ip-adapter_clip_sd15",
        "call_function": functools.partial(clip, config='clip_h'),
        "unload_function": functools.partial(unload_clip, config='clip_h'),
        "managed_model": None,
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "IP-Adapter"
        ]
    },
    "ip-adapter_clip_sdxl": {
        "name": "ip-adapter_clip_sdxl",
        "call_function": functools.partial(clip, config='clip_g'),
        "unload_function": functools.partial(unload_clip, config='clip_g'),
        "managed_model": None,
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "IP-Adapter"
        ]
    },
    "ip-adapter_clip_sdxl_plus_vith": {
        "name": "ip-adapter_clip_sdxl_plus_vith",
        "call_function": functools.partial(clip, config='clip_h'),
        "unload_function": functools.partial(unload_clip, config='clip_h'),
        "managed_model": None,
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "IP-Adapter"
        ]
    },
    "ip-adapter_face_id": {
        "name": "ip-adapter_face_id",
        "call_function": g_insight_face_model.run_model,
        "unload_function": None,
        "managed_model": "g_insight_face_model",
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "IP-Adapter"
        ]
    },
    "ip-adapter_face_id_plus": {
        "name": "ip-adapter_face_id_plus",
        "call_function": face_id_plus,
        "unload_function": functools.partial(unload_clip, config='clip_h'),
        "managed_model": None,
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "IP-Adapter"
        ]
    },
    "lineart_anime": {
        "name": "lineart_anime",
        "call_function": lineart_anime,
        "unload_function": unload_lineart_anime,
        "managed_model": "model_lineart_anime",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_anime_denoise": {
        "name": "lineart_anime_denoise",
        "call_function": lineart_anime_denoise,
        "unload_function": unload_lineart_anime_denoise,
        "managed_model": "model_manga_line",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_coarse": {
        "name": "lineart_coarse",
        "call_function": lineart_coarse,
        "unload_function": unload_lineart_coarse,
        "managed_model": "model_lineart_coarse",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_realistic": {
        "name": "lineart_realistic",
        "call_function": lineart,
        "unload_function": unload_lineart,
        "managed_model": "model_lineart",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Lineart"
        ]
    },
    "lineart_standard (from white bg & black line)": {
        "name": "lineart_standard (from white bg & black line)",
        "call_function": lineart_standard,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Lineart"
        ]
    },
    "mediapipe_face": {
        "name": "mediapipe_face",
        "call_function": mediapipe_face,
        "unload_function": None,
        "managed_model": "model_mediapipe_face",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": {
            "name": "Max Faces",
            "value": 1,
            "min": 1,
            "max": 10,
            "step": 1
        },
        "slider_2": {
            "name": "Min Face Confidence",
            "value": 0.5,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01
        },
        "slider_3": None,
        "priority": 0,
        "tags": []
    },
    "mlsd": {
        "name": "mlsd",
        "call_function": mlsd,
        "unload_function": unload_mlsd,
        "managed_model": "model_mlsd",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": {
            "name": "MLSD Value Threshold",
            "min": 0.01,
            "max": 2.0,
            "value": 0.1,
            "step": 0.01
        },
        "slider_2": {
            "name": "MLSD Distance Threshold",
            "min": 0.01,
            "max": 20.0,
            "value": 0.1,
            "step": 0.01
        },
        "slider_3": None,
        "priority": 100,
        "tags": [
            "MLSD"
        ]
    },
    # "normal_bae": {
    #     "name": "normal_bae",
    #     "call_function": normal_bae,
    #     "unload_function": unload_normal_bae,
    #     "managed_model": "model_normal_bae",
    #     "model_free": False,
    #     "no_control_mode": False,
    #     "resolution": None,
    #     "slider_1": None,
    #     "slider_2": None,
    #     "slider_3": None,
    #     "priority": 100,
    #     "tags": [
    #         "NormalMap"
    #     ]
    # },
    "normal_midas": {
        "name": "normal_midas",
        "call_function": midas_normal,
        "unload_function": unload_midas,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": {
            "name": "Normal Background Threshold",
            "min": 0.0,
            "max": 1.0,
            "value": 0.4,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "NormalMap"
        ]
    },
    "openpose": {
        "name": "openpose",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=False),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_face": {
        "name": "openpose_face",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=False, include_face=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_faceonly": {
        "name": "openpose_faceonly",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=False, include_hand=False, include_face=True),
        "unload_function": None,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_full": {
        "name": "openpose_full",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=True),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "OpenPose"
        ]
    },
    "openpose_hand": {
        "name": "openpose_hand",
        "call_function": functools.partial(g_openpose_model.run_model, include_body=True, include_hand=True, include_face=False),
        "unload_function": g_openpose_model.unload,
        "managed_model": "g_openpose_model",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "OpenPose"
        ]
    },
    "recolor_intensity": {
        "name": "recolor_intensity",
        "call_function": recolor_intensity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Gamma Correction",
            "value": 1.0,
            "min": 0.1,
            "max": 2.0,
            "step": 0.001
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Recolor"
        ]
    },
    "recolor_luminance": {
        "name": "recolor_luminance",
        "call_function": recolor_luminance,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Gamma Correction",
            "value": 1.0,
            "min": 0.1,
            "max": 2.0,
            "step": 0.001
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Recolor"
        ]
    },
    "reference_adain": {
        "name": "reference_adain",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": True,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Style Fidelity (only for Balanced mode)",
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Reference"
        ]
    },
    "reference_adain+attn": {
        "name": "reference_adain+attn",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": True,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Style Fidelity (only for Balanced mode)",
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Reference"
        ]
    },
    "reference_only": {
        "name": "reference_only",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": True,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Style Fidelity (only for Balanced mode)",
            "value": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Reference"
        ]
    },
    "revision_clipvision": {
        "name": "revision_clipvision",
        "call_function": functools.partial(clip, config='clip_g'),
        "unload_function": functools.partial(unload_clip, config='clip_g'),
        "managed_model": None,
        "model_free": True,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": {
            "name": "Noise Augmentation",
            "value": 0.0,
            "min": 0.0,
            "max": 1.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Revision"
        ]
    },
    "revision_ignore_prompt": {
        "name": "revision_ignore_prompt",
        "call_function": functools.partial(clip, config='clip_g'),
        "unload_function": functools.partial(unload_clip, config='clip_g'),
        "managed_model": None,
        "model_free": True,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": {
            "name": "Noise Augmentation",
            "value": 0.0,
            "min": 0.0,
            "max": 1.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Revision"
        ]
    },
    "scribble_hed": {
        "name": "scribble_hed",
        "call_function": scribble_hed,
        "unload_function": None,
        "managed_model": "model_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Scribble", "Sketch",
        ]
    },
    "scribble_pidinet": {
        "name": "scribble_pidinet",
        "call_function": scribble_pidinet,
        "unload_function": None,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Scribble", "Sketch",
        ]
    },
    "scribble_xdog": {
        "name": "scribble_xdog",
        "call_function": scribble_xdog,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": {
            "name": "XDoG Threshold",
            "min": 1,
            "max": 64,
            "value": 32
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Scribble", "Sketch",
        ]
    },
    "seg_anime_face": {
        "name": "seg_anime_face",
        "call_function": anime_face_segment,
        "unload_function": unload_anime_face_segment,
        "managed_model": "model_anime_face_segment",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Segmentation"
        ]
    },
    "seg_ofade20k": {
        "name": "seg_ofade20k",
        "call_function": oneformer_ade20k,
        "unload_function": unload_oneformer_ade20k,
        "managed_model": "model_oneformer_ade20k",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Segmentation"
        ]
    },
    "seg_ofcoco": {
        "name": "seg_ofcoco",
        "call_function": oneformer_coco,
        "unload_function": unload_oneformer_coco,
        "managed_model": "model_oneformer_coco",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Segmentation"
        ]
    },
    "seg_ufade20k": {
        "name": "seg_ufade20k",
        "call_function": uniformer,
        "unload_function": unload_uniformer,
        "managed_model": "model_uniformer",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Segmentation"
        ]
    },
    "shuffle": {
        "name": "shuffle",
        "call_function": shuffle,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Shuffle"
        ]
    },
    "softedge_hed": {
        "name": "softedge_hed",
        "call_function": hed,
        "unload_function": unload_hed,
        "managed_model": "model_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "softedge_hedsafe": {
        "name": "softedge_hedsafe",
        "call_function": hed_safe,
        "unload_function": None,
        "managed_model": "model_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "min": 64,
            "max": 2048,
            "value": 512
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "softedge_pidinet": {
        "name": "softedge_pidinet",
        "call_function": pidinet,
        "unload_function": unload_pidinet,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "SoftEdge"
        ]
    },
    "softedge_pidisafe": {
        "name": "softedge_pidisafe",
        "call_function": pidinet_safe,
        "unload_function": None,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "softedge_teed": {
        "name": "softedge_teed",
        "call_function": te_hed,
        "unload_function": unload_te_hed,
        "managed_model": "model_te_hed",
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": {
            "name": "Safe Steps",
            "min": 0,
            "max": 10,
            "value": 2,
            "step": 1
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "SoftEdge"
        ]
    },
    "t2ia_color_grid": {
        "name": "t2ia_color_grid",
        "call_function": color,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "T2I-Adapter"
        ]
    },
    "t2ia_sketch_pidi": {
        "name": "t2ia_sketch_pidi",
        "call_function": pidinet_ts,
        "unload_function": None,
        "managed_model": "model_pidinet",
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "T2I-Adapter"
        ]
    },
    "t2ia_style_clipvision": {
        "name": "t2ia_style_clipvision",
        "call_function": functools.partial(clip, config='clip_vitl'),
        "unload_function": functools.partial(unload_clip, config='clip_vitl'),
        "managed_model": None,
        "model_free": False,
        "no_control_mode": True,
        "resolution": None,
        "slider_1": None,
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "T2I-Adapter"
        ]
    },
    "threshold": {
        "name": "threshold",
        "call_function": threshold,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": {
            "name": "Preprocessor Resolution",
            "value": 512,
            "min": 64,
            "max": 2048
        },
        "slider_1": {
            "name": "Binarization Threshold",
            "min": 0,
            "max": 255,
            "value": 127
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": []
    },
    "tile_colorfix": {
        "name": "tile_colorfix",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Variation",
            "value": 8.0,
            "min": 3.0,
            "max": 32.0,
            "step": 1.0
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Tile", "Blur",
        ]
    },
    "tile_colorfix+sharp": {
        "name": "tile_colorfix+sharp",
        "call_function": identity,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Variation",
            "value": 8.0,
            "min": 3.0,
            "max": 32.0,
            "step": 1.0
        },
        "slider_2": {
            "name": "Sharpness",
            "value": 1.0,
            "min": 0.0,
            "max": 2.0,
            "step": 0.01
        },
        "slider_3": None,
        "priority": 0,
        "tags": [
            "Tile", "Blur",
        ]
    },
    "tile_resample": {
        "name": "tile_resample",
        "call_function": tile_resample,
        "unload_function": None,
        "managed_model": None,
        "model_free": False,
        "no_control_mode": False,
        "resolution": None,
        "slider_1": {
            "name": "Down Sampling Rate",
            "value": 1.0,
            "min": 1.0,
            "max": 8.0,
            "step": 0.01
        },
        "slider_2": None,
        "slider_3": None,
        "priority": 100,
        "tags": [
            "Tile", "Blur",
        ]
    }
}
