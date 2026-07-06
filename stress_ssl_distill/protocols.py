from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


STRESS_STAGE_NAMES_V1 = [
    "Baseline",
    "Stroop",
    "First Rest",
    "TMCT",
    "Second Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Subtract",
]

STRESS_STAGE_NAMES_V2 = [
    "Baseline",
    "TMCT",
    "First Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Second Rest",
    "Subtract",
]


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    start_idx: int
    end_idx: int
    is_stress: bool
    keep_for_training: bool = True


SEGMENTS_V1: List[SegmentSpec] = [
    SegmentSpec("baseline", 0, 1, False, True),
    SegmentSpec("stroop", 3, 4, True, True),
    SegmentSpec("tmct", 5, 6, True, True),
    SegmentSpec("real_opinion", 7, 8, True, True),
    SegmentSpec("opposite_opinion", 9, 10, True, True),
    SegmentSpec("subtract", 11, 12, True, True),
]

SEGMENTS_V2: List[SegmentSpec] = [
    SegmentSpec("baseline", 0, 1, False, True),
    SegmentSpec("tmct", 2, 3, True, True),
    SegmentSpec("real_opinion", 4, 5, True, True),
    SegmentSpec("opposite_opinion", 6, 7, True, True),
    SegmentSpec("subtract", 8, 9, True, True),
]


def normalize_subject_id(subject_id: str) -> str:
    return subject_id.strip()


def protocol_version_from_subject(subject_id: str) -> str:
    subject_id = normalize_subject_id(subject_id)
    return "V1" if subject_id.startswith("S") else "V2"


def expected_segments(subject_id: str) -> List[SegmentSpec]:
    return SEGMENTS_V1 if protocol_version_from_subject(subject_id) == "V1" else SEGMENTS_V2


def stress_score_columns(version: str) -> List[str]:
    return STRESS_STAGE_NAMES_V1 if version == "V1" else STRESS_STAGE_NAMES_V2


def canonical_stage_to_score_key(version: str) -> Dict[str, str]:
    if version == "V1":
        return {
            "baseline": "Baseline",
            "stroop": "Stroop",
            "tmct": "TMCT",
            "real_opinion": "Real Opinion",
            "opposite_opinion": "Opposite Opinion",
            "subtract": "Subtract",
        }
    return {
        "baseline": "Baseline",
        "tmct": "TMCT",
        "real_opinion": "Real Opinion",
        "opposite_opinion": "Opposite Opinion",
        "subtract": "Subtract",
    }


def stress_segments_for_subject(subject_id: str) -> List[str]:
    return [seg.name for seg in expected_segments(subject_id) if seg.is_stress and seg.keep_for_training]

