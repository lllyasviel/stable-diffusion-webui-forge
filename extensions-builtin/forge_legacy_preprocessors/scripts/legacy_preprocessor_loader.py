# This is a python script to convert all old preprocessors to new format.
# However, the old preprocessors are not very memory effective
# and eventually we should move all old preprocessors to new format
# see also the forge_preprocessor_normalbae/scripts/preprocessor_normalbae for
# how to make better implementation of preprocessors.
# No newer preprocessors should be written in this legacy way.

from legacy_preprocessors.preprocessor_meta import cn_preprocessor_modules, cn_preprocessor_unloadable

a = 0
