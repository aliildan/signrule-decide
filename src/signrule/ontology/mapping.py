"""Loaders for the hand-reviewed ontology tables in configs/ontology/ (design-02 §5, §8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml

from signrule.common.paths import CONFIG_DIR
from signrule.ontology.cir import ROLES, Group

ONTOLOGY_DIR = CONFIG_DIR / "ontology"
Kind = Literal["rule", "statutory", "person", "procuration"]


@dataclass(frozen=True)
class NoRuleSpec:
    kind: Kind
    alternatives: tuple[Group, ...]
    mode: str | None
    description: str


def parse_group(obj: Any) -> Group:
    if isinstance(obj, dict):
        return Group(collective=obj["collective"])
    return Group(frozenset((str(role), int(n)) for role, n in obj))


@cache
def load_no_rules(path: Path = ONTOLOGY_DIR / "no_rules.yaml") -> dict[str, NoRuleSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = {}
    for code, spec in raw.items():
        out[str(code)] = NoRuleSpec(
            kind=spec["kind"],
            alternatives=tuple(parse_group(g) for g in spec.get("alternatives", [])),
            mode=spec.get("mode"),
            description=spec["description"],
        )
    return out


@dataclass(frozen=True)
class RoleInfo:
    cir: str
    label: str


@dataclass
class RolesConfig:
    codes: dict[str, dict[str, RoleInfo]]
    order: dict[str, list[str]]
    na: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)

    def info(self, jur: str, code: str, fallback_label: str | None = None) -> RoleInfo:
        known = self.codes.get(jur, {}).get(code)
        return known or RoleInfo("OTHER", fallback_label or code)

    def order_key(self, jur: str, code: str) -> tuple[int, str]:
        order = self.order.get(jur, [])
        return (order.index(code) if code in order else len(order), code)

    def not_applicable(self, jur: str, legal_form: str | None) -> frozenset[str]:
        return self.na.get(jur, {}).get(legal_form or "", frozenset())


@cache
def load_roles(path: Path = ONTOLOGY_DIR / "roles.yaml") -> RolesConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    codes, order, na = {}, {}, {}
    for jur, cfg in raw.items():
        if not isinstance(jur, str):
            raise ValueError(f"roles.yaml: quote jurisdiction keys (YAML reads NO as {jur!r})")
        codes[jur] = {c: RoleInfo(v["cir"], v["label"]) for c, v in cfg["codes"].items()}
        bad = {i.cir for i in codes[jur].values()} - ROLES
        if bad:
            raise ValueError(f"roles.yaml {jur}: unknown CIR roles {sorted(bad)}")
        order[jur] = list(cfg.get("order", []))
        na[jur] = {f: frozenset(qs) for f, qs in (cfg.get("not_applicable") or {}).items()}
    return RolesConfig(codes, order, na)


@cache
def load_legal_vocab(jur: str, path: Path = ONTOLOGY_DIR / "legal_vocab.yaml") -> frozenset[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if any(not isinstance(k, str) for k in raw):
        raise ValueError("legal_vocab.yaml: quote jurisdiction keys (YAML reads NO as False)")
    return frozenset(w.lower() for w in raw.get(jur, []))
