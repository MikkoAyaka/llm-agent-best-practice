import importlib
from typing import Any, Callable

import loguru


def soft_import(module: str, attr: str) -> Any:
    _module = importlib.import_module(module)
    return getattr(_module, attr)


def py_require(func: Callable = lambda: (), extra_failed_msg: str = ""):
    try:
        func()
    except ImportError as e:
        if len(extra_failed_msg) > 0:
            loguru.logger.warning(e.msg + " >> " + extra_failed_msg)
        else:
            loguru.logger.warning(e.msg)
