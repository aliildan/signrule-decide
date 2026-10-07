"""Dedup groups, split assignment and leakage checks (CLAUDE.md §6 steps 6-7, design-03 §2).

Unit of deduplication: (jurisdiction, legal form, signature text key, procuration text key,
roles signature). Unit of splitting: the *text cluster* (jurisdiction, signature key, procuration
key), so identical texts never straddle splits. Records without any text are clustered by their
dedup key instead, so the empty text does not become one giant cluster.

Split assignment hashes the cluster key, so it is stable while the data grows (an incremental
fetch never moves an existing cluster to another split).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from datasketch import MinHash, MinHashLSH


@dataclass
class Member:
    entity_id: str
    request: dict[str, Any]
    signature_key: str
    procuration_key: str
    registered: date | None
    has_text: bool


@dataclass
class DedupGroup:
    key: tuple[str, ...]
    cluster: str
    members: list[Member] = field(default_factory=list)
    conflicts: dict[str, int] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def n_duplicates(self) -> int:
        return len(self.members)

    @property
    def entity_ids(self) -> list[str]:
        return [m.entity_id for m in self.members]


def roles_signature(state: dict[str, Any]) -> str:
    return ";".join(f"{r['role']}:{r['count']}" for r in state.get("roles") or [])


def dedup_key(m: Member) -> tuple[str, ...]:
    st = m.request["state"]
    return (
        str(st.get("jurisdiction")),
        str(st.get("legal_form")),
        m.signature_key,
        m.procuration_key,
        roles_signature(st),
    )


def cluster_key(m: Member, key: tuple[str, ...]) -> str:
    parts = key[:1] + key[2:4] if m.has_text else key
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _merge_labels(group: DedupGroup) -> dict[str, Any]:
    """Majority label per question across members; ties drop the question (masked)."""
    base = group.members[0].request
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    raw: dict[str, dict[str, Any]] = defaultdict(dict)
    for m in group.members:
        for qid, q in m.request["questions"].items():
            k = repr(q["label"])
            votes[qid][k] += 1
            raw[qid][k] = q
    questions: dict[str, Any] = {}
    for qid, ctr in votes.items():
        (top, n1), *rest = ctr.most_common(2) + [("", 0)]
        n2 = rest[0][1] if rest else 0
        if len(ctr) > 1:
            group.conflicts[qid] = sum(ctr.values()) - n1
        if n1 == n2:
            continue  # tie: label undetermined -> masked
        questions[qid] = raw[qid][top]
    return {"state": base["state"], "questions": questions}


def dedup(members: Iterable[Member]) -> list[DedupGroup]:
    groups: dict[tuple[str, ...], DedupGroup] = {}
    for m in members:
        key = dedup_key(m)
        g = groups.get(key)
        if g is None:
            g = groups[key] = DedupGroup(key=key, cluster=cluster_key(m, key))
        g.members.append(m)
    out = []
    for g in groups.values():
        g.request = _merge_labels(g)
        if g.request["questions"]:
            out.append(g)
    return out


def hash_split(
    cluster: str, seed: int = 13, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> str:
    h = int(hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    return "val" if h < ratios[0] + ratios[1] else "test"


def random_split(groups: list[DedupGroup], seed: int = 13) -> dict[str, str]:
    """cluster -> split."""
    return {g.cluster: hash_split(g.cluster, seed) for g in groups}


def temporal_split(
    groups: list[DedupGroup], cutoff: date = date(2025, 1, 1), seed: int = 13
) -> tuple[dict[str, str], int]:
    """A cluster is test only if every entity in it registered on/after `cutoff` (texts first
    seen after the cutoff); everything else is train/val. Returns (cluster -> split, number of
    post-cutoff entities that reuse an older text and are therefore not tested)."""
    newest: dict[str, bool] = {}
    for g in groups:
        all_new = all(m.registered is not None and m.registered >= cutoff for m in g.members)
        newest[g.cluster] = newest.get(g.cluster, True) and all_new
    seen_text_new = sum(
        1
        for g in groups
        if not newest[g.cluster]
        for m in g.members
        if m.registered is not None and m.registered >= cutoff
    )
    out = {}
    for c, is_new in newest.items():
        out[c] = (
            "test"
            if is_new
            else ("val" if hash_split(c, seed, (0.9, 0.1, 0.0)) == "val" else "train")
        )
    return out, seen_text_new


@dataclass
class LeakageReport:
    entity_overlap: int
    exact_text_overlap: int
    near_dup_rate: float
    n_test_clusters: int

    @property
    def ok(self) -> bool:
        return self.entity_overlap == 0 and self.exact_text_overlap == 0


def _shingles(s: str) -> set[str]:
    s = f"  {s}  "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def leakage_report(
    groups: list[DedupGroup], assign: dict[str, str], threshold: float = 0.95
) -> LeakageReport:
    ent_split: dict[str, set[str]] = defaultdict(set)
    text_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for g in groups:
        sp = assign[g.cluster]
        for m in g.members:
            ent_split[m.entity_id].add(sp)
            if m.has_text:
                text_split[(m.signature_key, m.procuration_key)].add(sp)
    entity_overlap = sum(1 for s in ent_split.values() if len(s) > 1)
    exact = sum(1 for s in text_split.values() if "test" in s and "train" in s)

    def mh(text: str) -> MinHash:
        m = MinHash(num_perm=64, seed=1)
        for sh in _shingles(text):
            m.update(sh.encode())
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    train_texts = {k for k, s in text_split.items() if "train" in s}
    for i, (sk, pk) in enumerate(sorted(train_texts)):
        lsh.insert(f"t{i}", mh(f"{sk}|{pk}"))
    test_texts = [k for k, s in text_split.items() if s == {"test"}]
    near = sum(1 for sk, pk in test_texts if lsh.query(mh(f"{sk}|{pk}")))
    return LeakageReport(entity_overlap, exact, near / max(len(test_texts), 1), len(test_texts))
