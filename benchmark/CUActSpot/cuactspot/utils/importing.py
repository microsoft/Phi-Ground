from __future__ import annotations

from importlib import import_module


def load_symbol(target: str):
    module_name, separator, attribute_name = target.partition(":")
    if not separator:
        module_name, separator, attribute_name = target.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "Dynamic targets must look like 'package.module:ClassName' or 'package.module.ClassName'"
        )
    module = import_module(module_name)
    return getattr(module, attribute_name)