from typing import Union

from .architecture.DAT import DAT
from .architecture.face.codeformer import CodeFormer
from .architecture.face.gfpganv1_clean_arch import GFPGANv1Clean
from .architecture.face.restoreformer_arch import RestoreFormer
from .architecture.HAT import HAT
from .architecture.LaMa import LaMa
from .architecture.OmniSR.OmniSR import OmniSR
from .architecture.RRDB import RRDBNet as ESRGAN
from .architecture.SCUNet import SCUNet
from .architecture.SPSR import SPSRNet as SPSR
from .architecture.SRVGG import SRVGGNetCompact as RealESRGANv2
from .architecture.SwiftSRGAN import Generator as SwiftSRGAN
from .architecture.Swin2SR import Swin2SR
from .architecture.SwinIR import SwinIR

PyTorchSRModels = (
    RealESRGANv2,
    SPSR,
    SwiftSRGAN,
    ESRGAN,
    SwinIR,
    Swin2SR,
    HAT,
    OmniSR,
    SCUNet,
    DAT,
)
PyTorchSRModel = Union[
    RealESRGANv2,
    SPSR,
    SwiftSRGAN,
    ESRGAN,
    SwinIR,
    Swin2SR,
    HAT,
    OmniSR,
    SCUNet,
    DAT,
]


def is_pytorch_sr_model(model: object):
    return isinstance(model, PyTorchSRModels)


PyTorchFaceModels = (GFPGANv1Clean, RestoreFormer, CodeFormer)
PyTorchFaceModel = Union[GFPGANv1Clean, RestoreFormer, CodeFormer]


def is_pytorch_face_model(model: object):
    return isinstance(model, PyTorchFaceModels)


PyTorchInpaintModels = (LaMa,)
PyTorchInpaintModel = Union[LaMa]


def is_pytorch_inpaint_model(model: object):
    return isinstance(model, PyTorchInpaintModels)


PyTorchModels = (*PyTorchSRModels, *PyTorchFaceModels, *PyTorchInpaintModels)
PyTorchModel = Union[PyTorchSRModel, PyTorchFaceModel, PyTorchInpaintModel]


def is_pytorch_model(model: object):
    return isinstance(model, PyTorchModels)
