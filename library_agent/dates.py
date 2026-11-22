"""How dates are written for people: "Sat 21 Nov" and "Saturday 21 November 2026".

The weekday is worked out from the date itself, so the screens and the chatbot are always right
about which day it is. (``%-d`` is not used because it does not work on Windows.)
"""
from datetime import date


def short(d: date) -> str:
    """Sat 21 Nov"""
    return f"{d:%a} {d.day} {d:%b}"


def full(d: date) -> str:
    """Saturday 21 November 2026"""
    return f"{d:%A} {d.day} {d:%B %Y}"