
from datetime import datetime


def get_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_etc(total_num: int, curr_num: int, start_time: float, current_time: float) -> float:
    """
    Get the estimated time to completion
    """
    elapsed_time_per_num = (current_time - start_time) / curr_num
    remaining_num = total_num - curr_num
    eta = remaining_num * elapsed_time_per_num
    return eta


def time_to_str(time_sec: float, print_days: bool = False) -> str:
    """Converts time in seconds to a string in the format dd:hh:mm:ss."""
    if print_days:
        days = time_sec // (24 * 3600)
        time_sec %= 24 * 3600
    else:
        days = -1

    hours = time_sec // 3600
    time_sec %= 3600

    minutes = time_sec // 60
    time_sec %= 60

    if print_days:
        return f"{days:.0f}:{hours:02.0f}:{minutes:02.0f}:{time_sec:02.0f}"
    else:
        return f"{hours:.0f}:{minutes:02.0f}:{time_sec:02.0f}"
