from __future__ import annotations

from importlib import import_module
from typing import Any


def load_class(class_path: str) -> type[Any]:
    """Load a class from `package.module:ClassName` or `package.module.ClassName`."""
    if not class_path:
        raise ValueError("class_path is required.")
    module_name, sep, class_name = class_path.partition(":")
    if not sep:
        module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid class_path: {class_path}")
    module = import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise TypeError(f"class_path must point to a class: {class_path}")
    return cls
