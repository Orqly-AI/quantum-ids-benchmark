"""List bib entries that manuscript.tex never cites, and show their titles.

Useful for spotting already-collected references that could strengthen the
Related Work without new literature search.

Run from paper/:  python _citecheck.py
"""
import re

tex = open("manuscript.tex", encoding="utf-8").read()
bib = open("references.bib", encoding="utf-8").read()

cited = set()
for group in re.findall(r"\\cite\{([^}]*)\}", tex):
    for key in group.split(","):
        cited.add(key.strip())

entries = {}
for m in re.finditer(r"^@[a-z]+\{([^,]+),(.*?)(?=^@|\Z)", bib, re.M | re.S):
    key, body = m.group(1).strip(), m.group(2)
    t = re.search(r"title\s*=\s*[{\"](.+?)[}\"]\s*,\s*$", body, re.M | re.S)
    entries[key] = " ".join(t.group(1).split()) if t else "(no title parsed)"

uncited = sorted(set(entries) - cited)
print(f"cited: {len(cited)}   in bib: {len(entries)}   uncited: {len(uncited)}\n")
for k in uncited:
    print(f"  {k}\n      {entries[k][:120]}")
