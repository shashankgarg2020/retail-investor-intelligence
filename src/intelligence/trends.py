"""Period-over-period change (docs/blueprint.md §8.4). Pure functions, no database."""
from __future__ import annotations

from dataclasses import dataclass

MIN_VOLUME = 5  # below this, a change is noise and is not called a trend


@dataclass
class Change:
    current: int
    previous: int
    absolute: int
    percent: float | None   # None when there is no previous volume to divide by
    significant: bool       # enough volume to be worth a user's attention
    direction: str          # up | down | flat | new

    @property
    def display(self) -> str:
        if self.direction == "new":
            return f"+{self.absolute} (no posts in previous period)"
        if self.percent is None:
            return "no change"
        return f"{self.percent:+.0f}% ({self.absolute:+d})"


def period_change(current: int, previous: int, min_volume: int = MIN_VOLUME) -> Change:
    absolute = current - previous
    if previous == 0:
        # A percentage from zero is infinite. Report the absolute growth instead.
        direction = "new" if current > 0 else "flat"
        return Change(current, previous, absolute, None, current >= min_volume, direction)
    percent = absolute / previous * 100
    direction = "up" if absolute > 0 else "down" if absolute < 0 else "flat"
    return Change(current, previous, absolute, percent, max(current, previous) >= min_volume, direction)
