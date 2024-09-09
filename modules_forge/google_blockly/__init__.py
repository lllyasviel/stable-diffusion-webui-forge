"""
This file is created by an inference implementation for Google Blockly.

Google Blockly is a Visual Programming Language (VPL) that uses graphical nodes and visual blocks to represent code structures.
For more information, refer to the official Google Blockly documentation and examples:

- Google Blockly repository: https://github.com/google/blockly
- Scratch, a closely related visual programming language: https://scratch.mit.edu/
- Scratch examples: https://www.google.com/search?tbm=isch&q=Scratch
- Scratch overview: https://en.wikipedia.org/wiki/Scratch_(programming_language)

**IMPORTANT:**

- This code contains an `exec` call. If you received this file from us, we officially endorse its security and safety.
  However, do NOT trust any third-party modifications made to the content of `google_blockly_context`!

**CONTENT:**

The `google_blockly_context` file is compressed using GZIP to reduce repository size (usually making files 10x smaller).
When decompressed, the `google_blockly_context` consists of two components:

1. **Google Blockly Visual Programming Graph (`google_blockly_GB_graph`):**

   - The graph `google_blockly_GB_graph` is the preferred and only form of the work for making modifications. This graph is the
     source of the program. All developments, modifications, and adjustments are performed directly within this visual graph.
   - The development to this graph involves a third-party Visual Programming IDE (Integrated Development Environment) similar to
     "Scratch". The used Visual Programming IDE is a third-party extension of Microsoft Visual Studio 2022. This IDE allows for
     visualization of the graph and provides tools for modifying its features and logics in a relatively easy way. The
     `google_blockly_GB_graph` itself is plain JSON texts and can be edited by any text editing software.

2. **Python Runtime for Google Blockly Visual Programming Graph (`GoogleBlocklyRuntime`):**

   - This runtime is projected directly from the Google Blockly standards and specifications.
   - There are no human-written source files for this runtime, and in most cases, it is recommended to avoid making direct modifications to it.

**LICENSE:**

The decompressed content of `google_blockly_context` is protected by the AGPL V3 license.
If you modify the `google_blockly_context` content, all changes must be governed by the AGPL V3 license:

- AGPL V3 License: https://www.gnu.org/licenses/agpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software distributed under this license is provided "AS IS,"
without warranties or conditions of any kind, either express or implied. Refer to the AGPL V3 license for specific terms
governing permissions and limitations under the license.
"""


import os
import gzip
import importlib.util


current_dir = os.path.dirname(__file__)
module_suffix = ".pyz"


def initialization():
    for filename in os.listdir(current_dir):
        if not filename.endswith(module_suffix):
            continue

        module_name = filename[:-len(module_suffix)]
        module_package_name = __package__ + '.' + module_name
        dynamic_module = importlib.util.module_from_spec(importlib.util.spec_from_loader(module_package_name, loader=None))
        dynamic_module.__dict__['__file__'] = os.path.join(current_dir, module_name + '.py')
        dynamic_module.__dict__['__package__'] = module_package_name
        google_blockly_context = gzip.open(os.path.join(current_dir, filename), 'rb').read().decode('utf-8')
        exec(google_blockly_context, dynamic_module.__dict__)
        globals()[module_name] = dynamic_module

    return
