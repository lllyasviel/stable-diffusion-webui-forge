from modules_forge.shared import Preprocessor, PreprocessorParameter, preprocessor_dir, add_preprocessor

# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format manually
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

from scripts.preprocessor_compiled import legacy_preprocessors

a = 0
