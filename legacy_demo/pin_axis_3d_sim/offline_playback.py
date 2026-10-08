"""Wall-clock playback controls for the local, kinematic presentation demo."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OfflinePlayback:
    total_commands: int
    speed: float = 5.0
    index: int = 0
    state: str = "ready"
    credit: float = 0.0

    def __post_init__(self) -> None:
        if self.total_commands <= 0:
            raise ValueError("playback requires commands")
        self.set_speed(self.speed)

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed) or not 0.25 <= speed <= 20.0:
            raise ValueError("playback speed must be between 0.25 and 20")
        self.speed = float(speed)

    def play(self) -> None:
        if self.state != "complete":
            self.state = "playing"

    def pause(self) -> None:
        if self.state == "playing":
            self.state = "paused"
        self.credit = 0.0

    def reset(self) -> None:
        self.index = 0
        self.credit = 0.0
        self.state = "ready"

    def advance(self, elapsed: float) -> range:
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed time must be finite and non-negative")
        start = self.index
        if self.state == "playing":
            # Do not jump forward after a blocked window or a debugger pause.
            self.credit += min(elapsed, 0.1) * 60.0 * self.speed
            count = min(int(self.credit + 1e-9), self.total_commands - start)
            self.credit = max(0.0, self.credit - count)
            self.index += count
            if self.index == self.total_commands:
                self.state = "complete"
                self.credit = 0.0
        return range(start, self.index)
