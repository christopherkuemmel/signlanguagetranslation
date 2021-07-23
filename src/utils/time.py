import time
import math


def startTime() -> float:
    """Method computes current time as seconds.

    Returns:
        time (float): Current time in seconds since Epoch.
    """
    return time.time()


def asHMS(seconds: float) -> str:
    """Converts seconds to hours mins seconds.

    Args:
        seconds (float): Seconds to convert to H M S.

    Returns:
        time (str): String representation of H M S.
    """
    minutes, sseconds = math.floor(seconds / 60), seconds % 60
    hours, minutes = math.floor(minutes / 60), minutes % 60
    return f"{hours:3}h {minutes:2}m {sseconds:5.2f}s"


def remaining(start_time: float, current_percent: float) -> str:
    """"""
    now = time.time()
    elapsed_sec = now - start_time
    estimated_remaining_sec = (elapsed_sec / (current_percent)) - elapsed_sec
    return asHMS(estimated_remaining_sec)


def timeSince(start_time: float, percent: float) -> str:
    """Computes time (H M S) since given seconds and estimates remaining time.

    Args:
        start_time (float): Starting time in seconds.
        percent (float): Current percentage - range(0.0,1.0)

    Returns:
        time (str): Time since start and estimated remaining time as str.
    """
    now = time.time()
    elapsed_sec = now - start_time
    estimated_remaining_sec = (elapsed_sec / (percent)) - elapsed_sec
    return f"Time since start: {asHMS(elapsed_sec)}\tEstimated remaining: {asHMS(estimated_remaining_sec)}"
