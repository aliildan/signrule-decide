"""Credentials from the git-ignored secrets folder, read at runtime and never printed."""

from __future__ import annotations

from signrule.common.paths import REPO_ROOT

SECRETS = REPO_ROOT / "secrets"
KEY_ALIASES = ("API_KEY", "APIKEY", "KEY", "TOKEN")


def load_secret(name: str, key: str) -> str:
    """Value of `key` from SECRETS/<name>.env or SECRETS/<name>/.env, never printed.

    Falls back to common aliases (<NAME>_API_KEY, API_KEY, ...), to the only KEY=VALUE line, or
    to a file holding just the bare value; reports only which variable name was used.
    """
    candidates = [SECRETS / f"{name}.env", SECRETS / name / ".env"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise SystemExit(f"missing credentials file for {name!r} in the secrets folder")
    pairs: dict[str, str] = {}
    bare: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            pairs[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
        else:
            bare.append(line)
    rel = path.relative_to(REPO_ROOT)
    for k in (key, *(f"{name.upper()}_{a}" for a in KEY_ALIASES), *KEY_ALIASES):
        for have in pairs:
            if have.upper() == k.upper():
                print(f"credentials: {rel} variable {have}")
                return pairs[have]
    if len(pairs) == 1:
        (only,) = pairs
        print(f"credentials: {rel} variable {only} (only entry)")
        return pairs[only]
    if not pairs and len(bare) == 1:
        print(f"credentials: {rel} (bare value)")
        return bare[0]
    raise SystemExit(f"{rel}: can't tell which entry is {key}; variable names: {sorted(pairs)}")
