TypedStorage = None


def encode(*args):
    pass


class RestrictedUnpickler:
    pass


allowed_zip_names_re = None
data_pkl_re = None


def check_zip_filenames(filename, names):
    pass


def check_pt(filename, extra_handler):
    pass


def load(filename, *args, **kwargs):
    pass


def load_with_extra(filename, extra_handler=None, *args, **kwargs):
    pass


class Extra:
    pass


unsafe_torch_load = None
global_extra_handler = None
