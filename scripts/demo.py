"""Demo against a running server (plan-09 T5; German/Austrian demo plan-16 T3).

    uv run python scripts/demo.py [--url http://127.0.0.1:8300]
    uv run python scripts/demo.py --lang de [--markdown results/demo/demo-de.md]

All texts below are HAND-WRITTEN ILLUSTRATIONS, not register records and not training data.
The release model (C″, plan-13) was trained on Norway + Austria; the Danish example shows
behaviour on a register it has never seen (strictest thresholds, `jurisdiction_calibrated: false`),
which is why abstentions there are expected and useful.
"""

from __future__ import annotations

import argparse
import sys

import httpx

from signrule.format.request import load_questions
from signrule.normalize.normalize_at import applicable_at

NO_ROLES = [
    {"role": "Daglig leder", "count": 1},
    {"role": "Styrets leder", "count": 1},
    {"role": "Styremedlem", "count": 2},
]

EXAMPLES: list[tuple[str, dict]] = [
    (
        "NO standard: CEO alone",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "Daglig leder alene.",
            "roles": NO_ROLES,
        },
    ),
    (
        "NO chair + one member",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "Styrets leder og ett styremedlem i fellesskap.",
            "roles": NO_ROLES,
        },
    ),
    (
        "NO rule can't fit roles",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "To styremedlemmer i fellesskap.",
            "roles": [{"role": "Daglig leder", "count": 1}, {"role": "Styrets leder", "count": 1}],
        },
    ),
    (
        "NO variant wording",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "Daglig leder og styreleder hver for seg.",
            "roles": NO_ROLES,
        },
    ),
    ("NO no rule registered", {"jurisdiction": "NO", "legal_form": "AS", "roles": NO_ROLES}),
    (
        "NO with condition",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "Styrets leder alene. Ved avtaler over 1 000 000 kr "
            "kreves i tillegg ett styremedlem.",
            "roles": NO_ROLES,
        },
    ),
    (
        "NO name in text (masked)",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "rule_text": "Kari Fiktivsen alene.",
            "roles": [
                {"role": "Daglig leder", "count": 1, "name": "Kari Fiktivsen"},
                {"role": "Styrets leder", "count": 1},
            ],
        },
    ),
    (
        "DK (untrained)",
        {
            "jurisdiction": "DK",
            "legal_form": "ApS",
            "signature_rule": "Selskabet tegnes af en direktør alene eller af den "
            "samlede bestyrelse.",
            "roles": [{"role": "Direktør", "count": 1}, {"role": "Bestyrelsesmedlem", "count": 3}],
        },
    ),
    (
        "NO two alternatives, both joint",
        {
            "jurisdiction": "NO",
            "legal_form": "AS",
            "signature_rule": "Daglig leder og styrets leder i fellesskap eller to styremedlemmer "
            "i fellesskap.",
            "roles": NO_ROLES,
        },
    ),
    (
        "AT GmbH, two GF each alone",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 selbständig; "
            "Geschäftsführer [PERSON_2]: vertritt seit 15.06.2021 selbständig",
            "roles": [{"role": "Geschäftsführer", "count": 2}],
        },
    ),
    (
        "AT GmbH, GF jointly with a GF or a Prokurist",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 gemeinsam mit "
            "einem weiteren Geschäftsführer oder einem Prokuristen; Geschäftsführer [PERSON_2]: "
            "vertritt seit 01.03.2019 gemeinsam mit einem weiteren Geschäftsführer oder einem "
            "Prokuristen",
            "procuration_rule": "Prokurist [PERSON_3]: vertritt seit 01.01.2022 gemeinsam mit "
            "einem Geschäftsführer",
            "roles": [{"role": "Geschäftsführer", "count": 2}, {"role": "Prokurist", "count": 1}],
        },
    ),
    (
        "AT unusual wording (not in the phrase table)",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 gemeinsam mit "
            "dem Vorsitzenden des Aufsichtsrats",
            "roles": [{"role": "Geschäftsführer", "count": 1}],
        },
    ),
]

# Austrian examples in the register's standard wording (§§ 18 GmbHG, 71 AktG, UGB), written by hand.
GF2 = [{"role": "Geschäftsführer", "count": 2}]
EXAMPLES_DE: list[tuple[str, dict]] = [
    (
        "GmbH: ein Geschäftsführer, selbständig",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 selbständig",
            "roles": [{"role": "Geschäftsführer", "count": 1}],
        },
    ),
    (
        "GmbH: zwei Geschäftsführer, jeder selbständig",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 selbständig; "
            "Geschäftsführer [PERSON_2]: vertritt seit 15.06.2021 selbständig",
            "roles": GF2,
        },
    ),
    (
        "GmbH: zwei Geschäftsführer, gemeinsam",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 gemeinsam mit "
            "einem weiteren Geschäftsführer; Geschäftsführer [PERSON_2]: vertritt seit 01.03.2019 "
            "gemeinsam mit einem weiteren Geschäftsführer",
            "roles": GF2,
        },
    ),
    (
        "GmbH: Geschäftsführer gemeinsam mit einem weiteren oder einem Prokuristen",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 gemeinsam mit "
            "einem weiteren Geschäftsführer oder einem Prokuristen; Geschäftsführer [PERSON_2]: "
            "vertritt seit 01.03.2019 gemeinsam mit einem weiteren Geschäftsführer oder einem "
            "Prokuristen",
            "procuration_rule": "Prokurist [PERSON_3]: Gesamtprokura gemeinsam mit einem "
            "Geschäftsführer",
            "roles": [*GF2, {"role": "Prokurist", "count": 1}],
        },
    ),
    (
        "GmbH: unterschiedliche Befugnisse je Person",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.03.2019 selbständig; "
            "Geschäftsführer [PERSON_2]: vertritt seit 01.07.2022 gemeinsam mit einem weiteren "
            "Geschäftsführer",
            "roles": GF2,
        },
    ),
    (
        "AG: Vorsitzender selbständig, Mitglieder gemeinsam",
        {
            "jurisdiction": "AT",
            "legal_form": "AG",
            "signature_rule": "Vorsitzender des Vorstands [PERSON_1]: vertritt seit 01.01.2020 "
            "selbständig; Vorstandsmitglied [PERSON_2]: vertritt seit 01.01.2020 gemeinsam mit "
            "einem weiteren Vorstandsmitglied oder einem Prokuristen; Vorstandsmitglied "
            "[PERSON_3]: vertritt seit 01.01.2020 gemeinsam mit einem weiteren Vorstandsmitglied "
            "oder einem Prokuristen",
            "roles": [
                {"role": "Vorsitzender des Vorstands", "count": 1},
                {"role": "Vorstandsmitglied", "count": 2},
            ],
        },
    ),
    (
        "Genossenschaft: Vorstand nur gemeinsam",
        {
            "jurisdiction": "AT",
            "legal_form": "Genossenschaft",
            "signature_rule": "Obmann [PERSON_1]: vertritt seit 12.05.2018 gemeinsam mit einem "
            "weiteren Vorstandsmitglied; Obmann-Stellvertreter [PERSON_2]: vertritt seit "
            "12.05.2018 gemeinsam mit einem weiteren Vorstandsmitglied; Vorstandsmitglied "
            "[PERSON_3]: vertritt seit 12.05.2018 gemeinsam mit einem weiteren Vorstandsmitglied",
            "roles": [
                {"role": "Obmann", "count": 1},
                {"role": "Obmann-Stellvertreter", "count": 1},
                {"role": "Vorstandsmitglied", "count": 1},
            ],
        },
    ),
    (
        "OG: zwei Gesellschafter, jeder selbständig",
        {
            "jurisdiction": "AT",
            "legal_form": "OG",
            "signature_rule": "unbeschränkt haftender Gesellschafter [PERSON_1]: vertritt seit "
            "02.02.2017 selbständig; unbeschränkt haftender Gesellschafter [PERSON_2]: vertritt "
            "seit 02.02.2017 selbständig",
            "roles": [{"role": "unbeschränkt haftender Gesellschafter", "count": 2}],
        },
    ),
    (
        "GmbH: ungewöhnliche Formulierung",
        {
            "jurisdiction": "AT",
            "legal_form": "GmbH",
            "signature_rule": "Geschäftsführer [PERSON_1]: vertritt seit 01.01.2020 gemeinsam mit "
            "dem Vorsitzenden des Aufsichtsrats",
            "roles": [{"role": "Geschäftsführer", "count": 1}],
        },
    ),
]
NOTES_DE = {
    "GmbH: unterschiedliche Befugnisse je Person": "Hinweis: Die Frage meint das Amt („kann *ein* "
    "Geschäftsführer kraft Amtes allein vertreten?“). Hier darf nur [PERSON_1] allein vertreten; "
    "nach unserer Konvention ist die Frage offen, das Modell antwortet „ja“. Bekannte Grenze.",
}
SHOW_DE = {
    "ceo_alone": "ein Geschäftsführer allein",
    "two_ceos": "zwei Geschäftsführer gemeinsam",
    "ceo_with_prokurist": "Geschäftsführer + Prokurist",
    "chair_alone": "Vorsitzender allein",
    "chair_with_member": "Vorsitzender + ein Mitglied",
    "member_alone": "ein Mitglied allein",
    "two_board_members_jointly": "zwei Mitglieder gemeinsam",
    "min_signers": "Mindestzahl Unterschriften",
    "prokura_joint": "Gesamtprokura",
}
CHOICE_DE = {"all_board": "alle", "3+": "3 oder mehr"}


def questions_for(state: dict, show: dict[str, str]) -> dict[str, str]:
    """The questions that make sense for the company's offices (as in training: applicable_at)."""
    kept = applicable_at(state, {q: True for q in show})
    if "procuration_rule" not in state:
        kept.pop("prokura_joint", None)
    return {q: show[q] for q in show if q in kept}


def fmt_de(a: dict) -> str:
    if a.get("abstain"):
        return "– (keine Antwort)"
    if a["type"] == "noul":
        p = float(a["noul"])
        return f"{'ja' if p >= 0.5 else 'nein'} ({max(p, 1 - p):.2f})".replace(".", ",")
    p = max(a["probabilities"].values())
    return f"{CHOICE_DE.get(a['choice'], a['choice'])} ({p:.2f})".replace(".", ",")


def run_de(url: str, markdown: str | None, model: str | None = None) -> int:
    qc = load_questions()
    lines = [
        "Alle Beispiele sind von Hand geschrieben (keine Registerdaten). Zahlen in Klammern: "
        "kalibrierte Wahrscheinlichkeit; „keine Antwort“ = das Modell enthält sich (α = 2 %).",
        "",
    ]
    with httpx.Client(timeout=120) as client:
        for title, state in EXAMPLES_DE:
            show = questions_for(state, SHOW_DE)
            questions = {q: dict(qc.questions[q]) for q in show}
            r = client.post(f"{url}/v1/systemone", json=payload(state, questions, model))
            r.raise_for_status()
            ans = r.json()["answers"]
            lines.append(f"### {title}")
            lines.append("")
            if title in NOTES_DE:
                lines += [f"_{NOTES_DE[title]}_", ""]
            lines.append(f"> {state['signature_rule']}")
            if state.get("procuration_rule"):
                lines.append(f"> {state['procuration_rule']}")
            lines.append("")
            lines.append("| Frage | Antwort |")
            lines.append("|---|---|")
            lines += [f"| {label} | {fmt_de(ans[q])} |" for q, label in show.items() if q in ans]
            lines.append("")
    text = "\n".join(lines)
    print(text)
    if markdown:
        from pathlib import Path

        Path(markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown).write_text(text + "\n", encoding="utf-8")
    return 0


SHOW = (
    "rule_type",
    "min_signers",
    "ceo_alone",
    "chair_alone",
    "two_ceos",
    "ceo_with_prokurist",
    "prokura_present",
)


def payload(state: dict, questions: dict, model: str | None) -> dict:
    """The /v1/systemone body; Ollama needs the model name, the reference server ignores it."""
    body = {"state": state, "questions": questions}
    return {"model": model, **body} if model else body


def fmt(a: dict) -> str:
    mark = "?" if a.get("abstain") else ""
    if a["type"] == "noul":
        return f"{'yes' if a['noul'] >= 0.5 else 'no'} {a['noul']:.2f}{mark}"
    p = max(a["probabilities"].values())
    return f"{a['choice']} {p:.2f}{mark}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://127.0.0.1:8300")
    ap.add_argument("--lang", choices=["en", "de"], default="en")
    ap.add_argument("--markdown", help="--lang de: also write the tables to this file")
    ap.add_argument("--model", help="model name, e.g. signrule-decide:4b-q8 for a local Ollama")
    a = ap.parse_args(argv)
    if a.lang == "de":
        return run_de(a.url, a.markdown, a.model)
    qc = load_questions()
    questions = {q: {k: v for k, v in qc.questions[q].items()} for q in SHOW}
    print("(? = abstains at α = 2 %; values are calibrated probabilities)\n")
    with httpx.Client(timeout=120) as client:
        for title, state in EXAMPLES:
            r = client.post(f"{a.url}/v1/systemone", json=payload(state, questions, a.model))
            r.raise_for_status()
            body = r.json()
            ans = body["answers"]
            seen = (
                ""
                if body.get("jurisdiction_calibrated", True)
                else "  [unseen register: strictest thresholds]"
            )
            print(f"■ {title}{seen}")
            print("   " + " | ".join(f"{q}: {fmt(ans[q])}" for q in SHOW if q in ans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
