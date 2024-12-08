import importlib
import os
from datetime import datetime
from typing import Callable, List

import inject


def realtime_tool_func(unit: str) -> int:
    """
    get the real time date,such as year 2024, month 12, day 15, day 2 of this week
    Args:
        unit(str): the time unit of real time date.
    Example argument inputs:
        'YEAR'
        'MONTH'
        'MINUTE'
        'HOUR'
        'DAY_OF_MONTH'
        'DAY_OF_WEEK'
    """
    now = datetime.now()
    if unit == 'MINUTE':
        return now.minute
    elif unit == 'YEAR':
        return now.year
    elif unit == 'MONTH':
        return now.month
    elif unit == 'HOUR':
        return now.hour
    elif unit == 'DAY_OF_MONTH':
        return now.day
    elif unit == 'DAY_OF_WEEK':
        return now.isoweekday()
    else:
        raise ValueError("Unsupported time unit")


@inject.autoparams()
def default_tool_kits() -> List[Callable]:
    tool_kits = [realtime_tool_func]

    return tool_kits
