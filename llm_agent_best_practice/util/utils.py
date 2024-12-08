import importlib
from datetime import datetime
from typing import Any, Callable

import loguru


def extract_history(history: dict) -> dict:
    print(history.keys())
    time = history.get("timestamp")
    prompt = history.get("prompt")
    messages = history.get("messages")
    response = history.get("response")
    return dict(time=time, prompt=prompt, message=messages, response=response)


def soft_import(module: str, attr: str) -> Any:
    _module = importlib.import_module(module)
    return getattr(_module, attr)


def py_require(func: Callable = lambda: (), extra_failed_msg: str = "") -> bool:
    try:
        func()
        return True
    except ImportError as e:
        if len(extra_failed_msg) > 0:
            loguru.logger.warning(e.msg + " >> " + extra_failed_msg)
        else:
            loguru.logger.warning(e.msg)
        return False
