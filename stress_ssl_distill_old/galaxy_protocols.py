from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


CALM_SESSIONS = {
    "baseline",
    "meditation-1",
    "meditation-2",
    "rest-1",
    "rest-2",
    "rest-3",
    "rest-4",
    "rest-5",
}

STRESS_SESSIONS = {
    "tsst-prep",
    "tsst-speech",
    "ssst-prep",
    "ssst-sing",
}

AMBIGUOUS_OR_NUISANCE_SESSIONS = {
    "adaptation",
    "screen-reading",
    "keyboard-typing",
    "mobile-typing",
    "standing",
    "walking",
    "jogging",
    "running",
}


@dataclass(frozen=True)
class EventInterval:
    session: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def session_to_label(session: str) -> Optional[int]:
    if session in CALM_SESSIONS:
        return 0
    if session in STRESS_SESSIONS:
        return 1
    return None


def session_to_group(session: str) -> str:
    if session in CALM_SESSIONS:
        return "calm"
    if session in STRESS_SESSIONS:
        return "stress"
    if session in AMBIGUOUS_OR_NUISANCE_SESSIONS:
        return "exclude"
    return "unknown"


def pair_event_intervals(event_rows: Iterable[Dict[str, object]]) -> List[EventInterval]:
    open_events: Dict[str, int] = {}
    intervals: List[EventInterval] = []

    for row in event_rows:
        session = str(row["session"]).strip()
        status = str(row["status"]).strip().upper()
        timestamp = int(row["timestamp"])

        if status == "ENTER":
            open_events[session] = timestamp
        elif status == "EXIT":
            start_ms = open_events.pop(session, None)
            if start_ms is None or timestamp <= start_ms:
                continue
            intervals.append(EventInterval(session=session, start_ms=start_ms, end_ms=timestamp))

    intervals.sort(key=lambda item: item.start_ms)
    return intervals
