"""Audit references.bib for metadata that is not submission-ready.

Flags placeholder authors, missing venue, missing DOI, and preprint-only
sources, separately for cited and uncited entries.

Run from paper/:  python _bibaudit.py
"""
import re

tex = open("manuscript.tex", encoding="utf-8").read()
bib = open("references.bib", encoding="utf-8").read()

cited = set()
for group in re.findall(r"\\cite\{([^}]*)\}", tex):
    cited.update(k.strip() for k in group.split(","))


def field(body, name):
    # trailing comma is optional: the last field of an entry has none
    m = re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S)
    return " ".join(m.group(1).split()) if m else None


rows = []
for m in re.finditer(r"^@([a-z]+)\{([^,]+),(.*?)(?=^@|\Z)", bib, re.M | re.S):
    kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
    author = field(body, "author") or ""
    problems = []
    if "Anonymous" in author or not author:
        problems.append("PLACEHOLDER AUTHOR")
    if not field(body, "journal") and not field(body, "booktitle") and kind != "misc":
        problems.append("no venue")
    if field(body, "journal") in ("arXiv preprint",):
        problems.append("preprint only")
    if kind == "misc":
        problems.append("misc/preprint")
    if not field(body, "doi"):
        problems.append("no DOI")
    rows.append((key, key in cited, problems))

for label, want in (("CITED IN MANUSCRIPT", True), ("NOT CITED", False)):
    print(f"\n===== {label} =====")
    for key, is_cited, problems in rows:
        if is_cited != want:
            continue
        flag = "  <-- BLOCKER" if any("PLACEHOLDER" in p for p in problems) else ""
        print(f"  {key:22s} {', '.join(problems) if problems else 'ok'}{flag}")
