from __future__ import annotations

import logging
import os

import torch

from modules import (
    devices,
    errors,
    face_restoration,
    face_restoration_utils,
    modelloader,
    shared,
)

logger = logging.getLogger(__name__)
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
model_download_name = "GFPGANv1.4.pth"
gfpgan_face_restorer: face_restoration.FaceRestoration | None = None


class FaceRestorerGFPGAN(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "GFPGAN"

    def get_device(self):
        return devices.device_gfpgan

    def load_net(self) -> torch.Module:
        for model_path in modelloader.load_models(
            model_path=self.model_path,
            model_url=model_url,
            command_path=self.model_path,
            download_name=model_download_name,
            ext_filter=['.pth'],
        ):
            if 'GFPGAN' in os.path.basename(model_path):
                return modelloader.load_spandrel_model(
                    model_path,
                    device=self.get_device(),
                    expected_architecture='GFPGAN',
                ).model

        #   if reach here, model not found. previous code will download it iff there are no models in GFPGAN directory
        #   this will download it if the supporting models exist
        try:
            GFPGANmodel = modelloader.load_file_from_url(model_url, model_dir=self.model_path, file_name=model_download_name)
            return modelloader.load_spandrel_model(
                GFPGANmodel,
                device=self.get_device(),
                expected_architecture='GFPGAN',
            ).model
        except:
            raise ValueError("No GFPGAN model found")

    def restore(self, np_image):
        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, return_rgb=False)[0]

        return self.restore_with_helper(np_image, restore_face)


def gfpgan_fix_faces(np_image):
    if gfpgan_face_restorer:
        return gfpgan_face_restorer.restore(np_image)
    logger.warning("GFPGAN face restorer not set up")
    return np_image


def setup_model(dirname: str) -> None:
    global gfpgan_face_restorer

    try:
        face_restoration_utils.patch_facexlib(dirname)
        gfpgan_face_restorer = FaceRestorerGFPGAN(model_path=dirname)
        shared.face_restorers.append(gfpgan_face_restorer)
    except Exception:
        errors.report("Error setting up GFPGAN", exc_info=True)
