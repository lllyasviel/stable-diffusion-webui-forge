from modules_forge.initialization import initialize_forge

initialize_forge()

import inspect

from backend import memory_management


def GPU(func, **kwargs):
    def wrapper():
        print("Entering Forge Space GPU ...")
        memory_management.unload_all_models()
        func()
        print("Quiting Forge Space GPU ...")
    return wrapper


def convert_root_path():
    frame = inspect.currentframe().f_back
    caller_file = frame.f_code.co_filename
    caller_file = os.path.abspath(caller_file)
    result = os.path.join(os.path.dirname(caller_file), 'huggingface_space_mirror')
    return result + '/'
