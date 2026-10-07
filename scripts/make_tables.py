"""Generate the README / model-card tables from results/ (CLAUDE.md §8: tables are generated).

    uv run python scripts/make_tables.py           # rewrite the <!-- table:NAME --> blocks
    uv run python scripts/make_tables.py --check   # exit 1 if a block is stale (CI)

Blocks: `<!-- table:NAME -->` … `<!-- /table -->` in README.md and model_card.md. Every number is
read from a results JSON; nothing is typed by hand. Also writes results/tables/release.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DOCS = ("README.md", "model_card.md")
COALITIONS = (
    "member_alone",
    "two_ceos",
    "ceo_with_chair",
    "ceo_with_prokurist",
    "chair_with_member",
    "three_members",
    "prokurist_alone",
    "two_prokurists",
    "chair_with_deputy",
    "deputy_with_member",
    "ceo_with_two_members",
)
RELEASE = "noat-4b-v2"
NAMES = {
    RELEASE: "**SignRule-Decide 4B**",
    "phrase_at": "Phrase table (rules from the training labels)",
    "rules_at": "Keyword rules (selbständig / gemeinsam)",
    "enc_mmBERT-base": "mmBERT-base, fine-tuned (1,024 tokens)",
    "enc_xlm-roberta-large": "XLM-R-large, fine-tuned (512 tokens)",
    "rules_no": "Register rule descriptions (exact match)",
}


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def load(tag: str, model: str) -> dict[str, Any] | None:
    p = RESULTS / tag / f"{model}.test.json"
    return json.loads(p.read_text()) if p.exists() else None


def acc(pq: dict[str, Any], q: str) -> float | None:
    v = pq.get(q)
    return v["calibrated"]["accuracy"] if v else None


def coalition_mean(pq: dict[str, Any]) -> tuple[float | None, int]:
    rows = [
        (v["calibrated"]["accuracy"], v["calibrated"]["n"])
        for q, v in pq.items()
        if q in COALITIONS
    ]
    n = sum(k for _, k in rows)
    return (sum(a * k for a, k in rows) / n if n else None), n


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def at_gold() -> str:
    rep = json.loads((RESULTS / "plan16" / "at-report.json").read_text())["at:gold (both batches)"]
    rows = []
    for m in (RELEASE, "enc_mmBERT-base", "enc_xlm-roberta-large", "phrase_at", "rules_at"):
        r = load("at-random-to-at-gold", m)
        if r is None:
            continue
        pq = r["per_question"]
        cm, _ = coalition_mean(pq)
        rows.append(
            [
                NAMES.get(m, m),
                pct(acc(pq, "ceo_alone")),
                pct(acc(pq, "chair_alone")),
                pct(cm),
                pct(acc(pq, "min_signers")),
                pct(acc(pq, "rule_type")),
            ]
        )
    head = [
        "Austria, reference set (400)",
        "GF alone",
        "Chair alone",
        "Coalitions (11)",
        "Min. signers",
        "Rule type",
    ]
    note = (
        f'\n\nSignRule-Decide 4B on the same items: dangerous yes/no errors ("can sign" when the '
        f'answer is "cannot") {rep["dangerous"]} of {rep["yes_no_items"]} '
        f"({pct(rep['dangerous_rate'])}); at a 2 % risk target it answers "
        f"{pct(rep['alpha_2pct']['coverage'])} of the questions with an observed risk of "
        f"{pct(rep['alpha_2pct']['risk'])}."
    )
    return table(head, rows) + note


def at_structures() -> str:
    rep = json.loads((RESULTS / "plan16" / "at-report.json").read_text())["by_structure"]
    label = {
        "vorstand": "Vorstand (AG, Genossenschaft, Privatstiftung)",
        "partner": "Partnerships (OG, KG)",
        "gmbh_multi": "GmbH with several Geschäftsführer",
    }
    rows = [
        [
            label[k],
            pct(v["coalition_mean"]),
            pct(v["min_signers"]),
            f"{v['dangerous']} / {v['yes_no_items']}",
        ]
        for k, v in rep.items()
    ]
    return table(
        ["Structure (second batch, 200)", "Coalitions", "Min. signers", "Dangerous errors"], rows
    )


def other_registers() -> str:
    rows = []
    no = load("no-random-to-no-gold", RELEASE)
    if no:
        pq = no["per_question"]
        cm, _ = coalition_mean(pq)
        rows.append(
            [
                "Norway, texts the official interpreter could not read (250)",
                pct(acc(pq, "ceo_alone")),
                pct(cm),
                pct(acc(pq, "min_signers")),
                pct(acc(pq, "rule_type")),
                "trained",
            ]
        )
    dk = json.loads((RESULTS / "plan14" / "dk-report.json").read_text()).get(RELEASE)
    dkr = load("at+no-random-to-dk-gold", RELEASE)
    if dk and dkr:
        pq = dkr["per_question"]
        rows.append(
            [
                "Denmark (300)",
                pct(acc(pq, "ceo_alone")),
                pct(dk["coalition_mean"]),
                pct(dk["min_signers"]),
                pct(dk["rule_type"]),
                "**never seen**",
            ]
        )
    return table(
        [
            "Reference set",
            "CEO alone",
            "Coalitions",
            "Min. signers",
            "Rule type",
            "Register in training?",
        ],
        rows,
    )


def in_distribution() -> str:
    rows = []
    for tag, label in (
        ("no-random", "Norway, held-out texts"),
        ("at-random", "Austria, held-out patterns"),
        ("at-random-to-at-pilot", "Austria, frozen pilot (2,500 companies)"),
    ):
        r = load(tag, RELEASE)
        if r is None:
            continue
        pq = r["per_question"]
        cm, _ = coalition_mean(pq)
        rows.append([label, pct(cm), pct(acc(pq, "min_signers")), pct(acc(pq, "rule_type"))])
    return table(
        ["Register-labelled test (codes / interpreter)", "Coalitions", "Min. signers", "Rule type"],
        rows,
    )


def data() -> str:
    d = json.loads((RESULTS / "data.json").read_text())
    no, at, dk, tr = d["NO"], d["AT"], d["DK"], d["training"]
    rows = [
        [
            "Norway",
            f"{no['entities_read']:,} companies ({no['api_responses']:,} API responses)",
            f"{no['entities_kept']:,}",
            f"{no['unique_groups']:,}",
            f"{no['train']:,} / {no['val']:,} / {no['test']:,}",
        ],
        [
            "Austria",
            f"{at['extracts_read']:,} company extracts",
            f"{at['companies_kept']:,}",
            f"{at['unique_patterns']:,} patterns",
            f"{at['train']:,} / {at['val']:,} / {at['test']:,} (+ {at['pilot']:,} pilot)",
        ],
        [
            "Denmark",
            f"{dk['companies_read']:,} companies",
            f"{dk['companies_kept']:,}",
            f"{dk['unique_texts']:,}",
            "evaluation only",
        ],
        [
            "**Total**",
            f"**{d['total_companies_read']:,} companies**",
            "",
            "",
            f"**{tr['training_cases']:,} training cases**, {tr['epochs']} epochs, "
            f"{tr['forward_tokens'] / 1e6:.1f} M tokens",
        ],
    ]
    return table(
        [
            "Register",
            "Read from the register",
            "Kept",
            "Unique (text, roles)",
            "Train / val / test",
        ],
        rows,
    )


def compute() -> str:
    c = json.loads((RESULTS / "compute.json").read_text())
    hw, tr, sv = c["hardware"], c["training"], c["serving"]
    rows = [
        ["GPU", f"{hw['gpu']['count']} × {hw['gpu']['name']} ({hw['gpu']['memory_gb']} GB)"],
        ["CPU / RAM", f"{hw['cpu']} / {hw['ram_gb']} GB"],
        [
            f"Training ({'LoRA' if tr['weights'] == 'lora' else tr['weights']}, {tr['precision']})",
            f"{tr['wall_hours']} h, peak GPU memory {tr['peak_gpu_memory_gb']} GB, "
            f"{tr['optimizer_steps']:,} optimizer steps",
        ],
        [
            f"Serving (one request, {sv['questions_per_request']:.1f} questions)",
            f"p50 {sv['p50_ms']} ms, p95 {sv['p95_ms']} ms, {sv['throughput_rps']} requests/s",
        ],
    ]
    return table(["Compute", "Value"], rows)


def _tex(cell: str) -> str:
    cell = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", cell.strip())
    cell = cell.replace(" %", "\\,\\%").replace("&", "\\&").replace("_", "\\_")
    return cell.replace('"', "''")


def to_latex(md: str) -> str:
    """A generated markdown table (plus trailing note) as a booktabs tabular."""
    lines = md.strip().split("\n")
    rows = [line for line in lines if line.startswith("|")]
    rest = [line for line in lines if not line.startswith("|") and line.strip()]
    cells = [[c for c in r.strip("|").split("|")] for r in rows if not set(r) <= set("|-")]
    head, body = cells[0], cells[1:]
    out = [f"\\begin{{tabular}}{{l{'r' * (len(head) - 1)}}}", "\\toprule"]
    out.append(" & ".join(_tex(c) for c in head) + " \\\\")
    out.append("\\midrule")
    out += [" & ".join(_tex(c) for c in r) + " \\\\" for r in body]
    out += ["\\bottomrule", "\\end{tabular}"]
    if rest:
        out += ["", " ".join(_tex(x) for x in rest)]
    return "\n".join(out)


TABLES = {
    "at-gold": at_gold,
    "at-structures": at_structures,
    "other-registers": other_registers,
    "in-distribution": in_distribution,
    "compute": compute,
    "data": data,
}
BLOCK = re.compile(r"(<!-- table:([\w-]+) -->\n)(.*?)(<!-- /table -->)", re.S)


def inject(doc: str, tables: dict[str, str]) -> tuple[str, bool]:
    def repl(m: re.Match[str]) -> str:
        name = m.group(2)
        return m.group(1) + (tables[name] + "\n" if name in tables else m.group(3)) + m.group(4)

    new = BLOCK.sub(repl, doc)
    return new, new != doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    tables = {name: fn() for name, fn in TABLES.items()}
    stale = []
    for doc in DOCS:
        path = REPO / doc
        if not path.exists():
            continue
        new, changed = inject(path.read_text(encoding="utf-8"), tables)
        if changed:
            stale.append(doc)
            if not a.check:
                path.write_text(new, encoding="utf-8")
    release = "\n\n".join(f"## {name}\n\n{t}" for name, t in tables.items()) + "\n"
    latex = {name: to_latex(t) + "\n" for name, t in tables.items()}
    out = RESULTS / "tables" / "release.md"
    if a.check:
        if stale or not out.exists() or out.read_text(encoding="utf-8") != release:
            print(f"make_tables: stale {stale or ['results/tables/release.md']}", file=sys.stderr)
            return 1
        print("make_tables: up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(release, encoding="utf-8")
    for name, tex in latex.items():
        (out.parent / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(f"make_tables: wrote {len(tables)} tables; updated {stale or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
