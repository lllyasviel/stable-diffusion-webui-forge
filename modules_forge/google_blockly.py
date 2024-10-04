# See also: https://github.com/lllyasviel/google_blockly_prototypes/blob/main/LICENSE_pyz


import os
import gzip
import importlib.util


pyz_dir = os.path.abspath(os.path.realpath(os.path.join(__file__, '../../repositories/google_blockly_prototypes/forge')))
module_suffix = ".pyz"


def initialization():
    print('Loading additional modules ... ', end='')

    for filename in os.listdir(pyz_dir):
        if not filename.endswith(module_suffix):
            continue

        module_name = filename[:-len(module_suffix)]
        module_package_name = __package__ + '.' + module_name
        dynamic_module = importlib.util.module_from_spec(importlib.util.spec_from_loader(module_package_name, loader=None))
        dynamic_module.__dict__['__file__'] = os.path.join(pyz_dir, module_name + '.py')
        dynamic_module.__dict__['__package__'] = module_package_name
        google_blockly_context = gzip.open(os.path.join(pyz_dir, filename), 'rb').read().decode('utf-8')
        exec(google_blockly_context, dynamic_module.__dict__)
        globals()[module_name] = dynamic_module

    print('done.')
    return
