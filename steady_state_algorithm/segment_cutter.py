"""
Turn a boolean steady/transient label array into a list of segments.

Rule: a new segment starts whenever the trace goes steady -> transient
again (i.e. a new transient run begins after a steady run has already
been seen). Each segment is [start_of_transient, end_of_following_steady].
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start_idx: int
    end_idx: int # inclusive
    transient_idx: "list[int]"
    steady_idx: "list[int]"


def cut_segments(steady: "list[bool]") -> List[Segment]:
    n = len(steady)
    segments = []

    seg_start = 0
    seen_steady_since_seg_start = False
    prev = steady[0]

    for i in range(1, n):
        cur = steady[i]
        if seen_steady_since_seg_start and bool(prev) and not bool(cur):
            # steady -> transient transition: close current segment here,
            # start a new one at i
            segments.append(_build_segment(seg_start, i - 1, steady))
            seg_start = i
            seen_steady_since_seg_start = False
        if cur:
            seen_steady_since_seg_start = True
        prev = cur

    # close final segment
    segments.append(_build_segment(seg_start, n - 1, steady))
    return segments


def _build_segment(start_idx, end_idx, steady) -> Segment:
    transient_idx = [i for i in range(start_idx, end_idx + 1) if not steady[i]]
    steady_idx = [i for i in range(start_idx, end_idx + 1) if steady[i]]
    return Segment(start_idx, end_idx, transient_idx, steady_idx)

    # steady_idx = [i for i in range(start_idx, end_idx + 1) if steady[i]]

    # if len(steady_idx) > 0:
    #     first_steady = steady_idx[0]
    #     transient_idx = list(range(start_idx, first_steady))
    # else:
    #     transient_idx = list(range(start_idx, end_idx + 1))

    # return Segment(start_idx, end_idx, transient_idx, steady_idx)

def build_true_segments(boundaries, steady):
    segments = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segments.append(
            Segment(
                start_idx=start,
                end_idx=end,
                transient_idx=[
                    i for i in range(start, end + 1)
                    if not steady[i]
                ],
                steady_idx=[
                    i for i in range(start, end + 1)
                    if steady[i]
                ],
            )
        )

    return segments