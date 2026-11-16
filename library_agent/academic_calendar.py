"""Academic calendar helper.

Given any date, say where it falls in the university year. The simulator uses
this to create realistic borrowing waves, and the demand forecaster uses it as
a set of *known-in-advance* features (we always know when exams are coming).

The file is not called ``calendar.py`` on purpose: that would shadow Python's
built-in ``calendar`` module.
"""
from dataclasses import dataclass
from datetime import date

from .config import BREAK_WEEKS, CALENDAR_START, EXAM_WEEKS, TERM_WEEKS

CYCLE_WEEKS = TERM_WEEKS + BREAK_WEEKS


@dataclass(frozen=True)
class Phase:
    week_of_term: int  # 1..TERM_WEEKS, or 0 during a break
    is_exam: bool
    is_break: bool


def phase(d: date) -> Phase:
    """Return the academic phase for date ``d``."""
    position = ((d - CALENDAR_START).days // 7) % CYCLE_WEEKS
    if position >= TERM_WEEKS:
        return Phase(week_of_term=0, is_exam=False, is_break=True)
    week = position + 1
    return Phase(week_of_term=week, is_exam=week > TERM_WEEKS - EXAM_WEEKS, is_break=False)