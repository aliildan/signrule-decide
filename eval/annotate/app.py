"""Local annotation tool for the gold sets (plan-05 T5). Follows docs/annotation_guidelines.md.

    uv run python eval/annotate/app.py --set no-rt --annotator ali     # http://127.0.0.1:8400

Shows one item at a time exactly as the model sees it (masked state), collects the signing rule as
alternatives of groups, procuration, ambiguity and a note, and appends to
data/gold/<set>/<annotator>.jsonl (latest entry per item wins). Annotators never see each other's
labels. Binds to localhost only.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from signrule.common.paths import GOLD_DIR
from signrule.ontology.cir import ROLES

N_GROUPS, N_ROLES = 4, 3
COLLECTIVES = ("", "ALL_BOARD", "ALL_PARTNERS", "ALL_EXECUTIVES")
ROLE_CHOICES = ("", *sorted(ROLES))

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}}
.state{{background:#f6f6f6;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:12px 0}}
.rule{{font-size:1.15em;font-weight:600}} table{{border-collapse:collapse}} td,th{{padding:2px 8px}}
fieldset{{margin:10px 0;border:1px solid #ccc;border-radius:6px}} .grp{{display:inline-block;
vertical-align:top;margin:4px 12px 4px 0;padding:6px;border:1px dashed #bbb;border-radius:6px}}
button{{padding:6px 16px;margin-right:8px}} .muted{{color:#777}}
</style></head><body>{body}</body></html>"""


def load_batch(set_name: str, root: Path) -> list[dict[str, Any]]:
    path = root / set_name / "batch.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_mine(set_name: str, annotator: str, root: Path) -> dict[str, dict[str, Any]]:
    path = root / set_name / f"{annotator}.jsonl"
    if not path.exists():
        return {}
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["item_id"]] = rec
    return latest


def form_to_record(form: dict[str, str], item_id: str, annotator: str) -> dict[str, Any]:
    if form.get("action") == "skip":
        return {"item_id": item_id, "annotator": annotator, "skipped": True}
    groups = []
    for g in range(N_GROUPS):
        coll = form.get(f"g{g}_collective", "")
        roles = []
        for r in range(N_ROLES):
            role, count = form.get(f"g{g}_r{r}_role", ""), form.get(f"g{g}_r{r}_count", "")
            if role in ROLES and count.strip().isdigit() and int(count) >= 1:
                roles.append([role, int(count)])
        if coll in COLLECTIVES[1:]:
            groups.append({"roles": [], "collective": coll})
        elif roles:
            groups.append({"roles": roles, "collective": None})
    status = form.get("status", "")
    if status not in ("rule", "uninterpretable", "none"):
        raise ValueError("choose a status")
    if status == "rule" and not groups:
        raise ValueError("status 'rule' needs at least one alternative group")
    amb = form.get("ambiguity", "")
    if amb not in ("0", "1", "2", "3"):
        raise ValueError("choose an ambiguity level")
    return {
        "item_id": item_id,
        "annotator": annotator,
        "skipped": False,
        "status": status,
        "alternatives": groups if status == "rule" else [],
        "person_specific": form.get("person_specific") == "on",
        "procuration": {
            "present": form.get("prok_present", "unknown"),
            "mode": form.get("prok_mode", "unknown"),
        },
        "ambiguity": int(amb),
        "note": form.get("note", "").strip(),
    }


def _sel(name: str, options: tuple[str, ...], current: str) -> str:
    opts = "".join(
        f'<option value="{o}"{" selected" if o == current else ""}>{o or "—"}</option>'
        for o in options
    )
    return f'<select name="{name}">{opts}</select>'


def _radio(name: str, options: tuple[str, ...], current: str) -> str:
    def one(o: str) -> str:
        checked = " checked" if o == current else ""
        return f'<label><input type="radio" name="{name}" value="{o}"{checked}> {o}</label>'

    return " ".join(one(o) for o in options)


def render_item(
    idx: int,
    item: dict[str, Any],
    total: int,
    done: int,
    prev: dict[str, Any] | None,
    error: str = "",
) -> str:
    st = item["state"] if isinstance(item["state"], dict) else {"text": item["state"]}
    e = html.escape
    roles = "".join(
        f"<tr><td>{e(str(r.get('role')))}</td><td>{e(str(r.get('count')))}</td></tr>"
        for r in st.get("roles") or []
    )
    notes = "".join(
        f"<li><b>{e(str(n.get('role')))}</b>: {e(str(n.get('note')))}</li>"
        for n in st.get("role_notes") or []
    )
    prev = prev or {}
    alts = prev.get("alternatives") or []
    groups_html = ""
    for g in range(N_GROUPS):
        alt = alts[g] if g < len(alts) else {}
        rows = alt.get("roles") or []
        inner = (
            "collective "
            + _sel(f"g{g}_collective", COLLECTIVES, alt.get("collective") or "")
            + "<br>"
        )
        for r in range(N_ROLES):
            role, cnt = rows[r] if r < len(rows) else ["", ""]
            inner += (
                _sel(f"g{g}_r{r}_role", ROLE_CHOICES, role)
                + f' × <input name="g{g}_r{r}_count" size="2" value="{cnt}"><br>'
            )
        groups_html += f'<div class="grp"><b>alternative {g + 1}</b> (AND)<br>{inner}</div>'
    prok = prev.get("procuration") or {}
    nav = (
        f'<a href="/item/{max(idx - 1, 0)}">◀ prev</a> · '
        f'<a href="/item/{min(idx + 1, total - 1)}">next ▶</a>'
    )
    err = f'<p style="color:#b00"><b>{e(error)}</b></p>' if error else ""
    status = _radio("status", ("rule", "uninterpretable", "none"), prev.get("status", ""))
    present = _radio("prok_present", ("yes", "no", "unknown"), prok.get("present", ""))
    mode = _radio("prok_mode", ("sole", "joint", "mixed", "unknown"), prok.get("mode", ""))
    amb = _radio("ambiguity", ("0", "1", "2", "3"), str(prev.get("ambiguity", "")))
    note = e(prev.get("note", ""))
    ps_checked = " checked" if prev.get("person_specific") else ""

    def lines(text: Any) -> str:
        parts = [p for p in str(text or "").split("; ") if p.strip()]
        if len(parts) <= 1:
            return e(str(text or "—"))
        return "<ul>" + "".join(f"<li>{e(p)}</li>" for p in parts) + "</ul>"

    body = f"""
<p class="muted">set <b>{e(item.get("set", ""))}</b> · item {idx + 1}/{total}
 · done {done} · {nav}</p>
{err}
<div class="state">
 <div>{e(str(st.get("jurisdiction", "")))} · {e(str(st.get("legal_form", "")))}</div>
 <div class="rule">Signing rule: {lines(st.get("signature_rule"))}</div>
 <div>Procuration rule: {lines(st.get("procuration_rule"))}</div>
 <table><tr><th>role</th><th>count</th></tr>{roles}</table>
 {f"<ul>{notes}</ul>" if notes else ""}
</div>
<form method="post">
 <fieldset><legend>Status</legend>{status}</fieldset>
 <fieldset><legend>Alternatives (OR) of groups (AND), guidelines §3</legend>
  {groups_html}</fieldset>
 <fieldset><legend>Person-specific powers (guidelines §6)</legend>
  <label><input type="checkbox" name="person_specific"{ps_checked}> some powers are registered
  for individual persons only, not for every holder of the office</label></fieldset>
 <fieldset><legend>Procuration</legend>present {present} · mode {mode}</fieldset>
 <fieldset><legend>Ambiguity (0 clear … 3 undecidable)</legend>{amb}</fieldset>
 <fieldset><legend>Note</legend>
  <textarea name="note" rows="2" cols="90">{note}</textarea></fieldset>
 <button name="action" value="save">Save &amp; next</button>
 <button name="action" value="skip">Skip</button>
</form>"""
    return PAGE.format(title=f"Annotate {idx + 1}/{total}", body=body)


def create_app(set_name: str, annotator: str, root: Path = GOLD_DIR) -> FastAPI:
    app = FastAPI(title="signrule-annotate")
    batch = load_batch(set_name, root)
    out_path = root / set_name / f"{annotator}.jsonl"

    def first_open() -> int:
        mine = load_mine(set_name, annotator, root)
        return next((i for i, it in enumerate(batch) if it["item_id"] not in mine), len(batch) - 1)

    @app.get("/")
    def index() -> RedirectResponse:
        return RedirectResponse(f"/item/{first_open()}", status_code=303)

    @app.get("/item/{idx}", response_class=HTMLResponse)
    def show(idx: int) -> str:
        mine = load_mine(set_name, annotator, root)
        item = batch[idx]
        return render_item(idx, item, len(batch), len(mine), mine.get(item["item_id"]))

    @app.post("/item/{idx}", response_class=HTMLResponse)
    async def save(idx: int, request: Request):  # noqa: ANN202
        form = {k: str(v) for k, v in (await request.form()).items()}
        item = batch[idx]
        try:
            rec = form_to_record(form, item["item_id"], annotator)
        except ValueError as err:
            mine = load_mine(set_name, annotator, root)
            return HTMLResponse(render_item(idx, item, len(batch), len(mine), None, str(err)), 422)
        rec["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return RedirectResponse(f"/item/{min(idx + 1, len(batch) - 1)}", status_code=303)

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=8400)
    a = ap.parse_args(argv)
    import uvicorn

    from signrule.common.paths import require_vault

    require_vault(GOLD_DIR)
    uvicorn.run(create_app(a.set, a.annotator), host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
