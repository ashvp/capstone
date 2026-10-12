"""Slot schema, hypothesis templating, single-slot interventions, compositional splits.

An *event* is a tuple of slot-value indices (action, object, scene, relation).
Because the schema is closed and small (8*6*6*6 = 1728 events), the entire
hypothesis space is enumerated once and every hypothesis sentence can be
embedded ahead of time ("text bank"). A *minimal intervention* is then exact:
exactly one slot index differs.

NOTE: the schema and ABNORMAL_RULES below are a PLACEHOLDER to be replaced by
your own surveillance-relevant schema after the pseudo-labelling pilot.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Optional, Set, Tuple

SLOT_NAMES = ("action", "object", "scene", "relation")

SCHEMA: Dict[str, List[str]] = {
    "action": ["takes", "places", "carries", "throws", "pushes", "hands over", "drops", "picks up"],
    "object": ["bag", "bottle", "box", "phone", "bicycle", "backpack"],
    "scene": ["store", "street", "corridor", "platform", "parking lot", "plaza"],
    "relation": [
        "from a shelf",
        "from another person's bag",
        "from the ground",
        "into their own bag",
        "to another person",
        "toward a person",
    ],
}

RADIX: Tuple[int, ...] = tuple(len(SCHEMA[s]) for s in SLOT_NAMES)
N_EVENTS: int = math.prod(RADIX)
N_SINGLE_EDITS: int = sum(r - 1 for r in RADIX)

TEMPLATE = "A person {action} a {object} {relation} in a {scene}."

# Placeholder normality prior over (action, relation). In the real project the
# normality head is trained from weak video-level labels; this table is used
# only (a) to generate synthetic data and (b) as an optional weak prior loss.
ABNORMAL_RULES: Set[Tuple[str, str]] = {
    ("takes", "from another person's bag"),
    ("throws", "toward a person"),
    ("pushes", "toward a person"),
}

Event = Tuple[int, int, int, int]


def encode(ev: Event) -> int:
    idx = 0
    for v, r in zip(ev, RADIX):
        idx = idx * r + int(v)
    return idx


def decode(idx: int) -> Event:
    out = []
    for r in reversed(RADIX):
        out.append(idx % r)
        idx //= r
    return tuple(reversed(out))  # type: ignore[return-value]


def all_events() -> List[Event]:
    """All events, ordered so that position i == encode(event)."""
    return list(itertools.product(*[range(r) for r in RADIX]))


def to_text(ev: Event) -> str:
    d = {name: SCHEMA[name][int(v)] for name, v in zip(SLOT_NAMES, ev)}
    return TEMPLATE.format(**d)


def is_abnormal(ev: Event) -> bool:
    return (SCHEMA["action"][ev[0]], SCHEMA["relation"][ev[3]]) in ABNORMAL_RULES


def single_slot_edits(ev: Event, slot: Optional[int] = None) -> List[Event]:
    """All events that differ from `ev` in exactly one slot (optionally a given slot)."""
    out: List[Event] = []
    slots = range(len(RADIX)) if slot is None else [slot]
    for j in slots:
        for v in range(RADIX[j]):
            if v != ev[j]:
                e = list(ev)
                e[j] = v
                out.append(tuple(e))  # type: ignore[arg-type]
    return out


def edit_slot_ids() -> List[int]:
    """Slot id of every entry returned by single_slot_edits(ev) (same order for all events)."""
    ids: List[int] = []
    for j, r in enumerate(RADIX):
        ids += [j] * (r - 1)
    return ids


def compositional_split(seed: int = 0, frac: float = 0.2) -> Set[Tuple[int, int]]:
    """Hold out (action, object) pairs, Something-Else style.

    Every action and every object keeps at least one *training* pair, so the
    held-out combinations are 'familiar factors in new combinations'.
    """
    rng = random.Random(seed)
    pairs = [(a, o) for a in range(RADIX[0]) for o in range(RADIX[1])]
    rng.shuffle(pairs)
    target = int(round(frac * len(pairs)))
    held: Set[Tuple[int, int]] = set()
    a_cnt = {a: RADIX[1] for a in range(RADIX[0])}
    o_cnt = {o: RADIX[0] for o in range(RADIX[1])}
    for a, o in pairs:
        if len(held) >= target:
            break
        if a_cnt[a] > 1 and o_cnt[o] > 1:
            held.add((a, o))
            a_cnt[a] -= 1
            o_cnt[o] -= 1
    return held


def is_heldout(ev: Event, held: Set[Tuple[int, int]]) -> bool:
    return (ev[0], ev[1]) in held
