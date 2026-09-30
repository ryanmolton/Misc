"""Lets ComfyUI load LumaLock when this whole `comfyui-colorize` folder is
copied into custom_nodes instead of just its `ComfyUI-LumaLock` subfolder."""

import importlib.util
import os
import sys

_pack = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI-LumaLock")
_name = "comfyui_lumalock_pack"
if _name in sys.modules:
    _mod = sys.modules[_name]
else:
    _spec = importlib.util.spec_from_file_location(
        _name, os.path.join(_pack, "__init__.py"), submodule_search_locations=[_pack])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)

NODE_CLASS_MAPPINGS = _mod.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _mod.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
