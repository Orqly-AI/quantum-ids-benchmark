"""Strip private annotations out of references.bib.

Two kinds of leak:

  * ``%`` comments. arXiv publishes the LaTeX source, so header lines such as
    the target venue and the internal "Tier A / Tier C" triage become publicly
    downloadable.
  * ``note = {...}`` fields. elsarticle-num *prints* the note field, so private
    reading notes ("concedes classical RF/XGBoost remain superior") end up
    inside the typeset reference list.

Run from paper/:  python _clean_bib.py [--write]
"""
import re
import sys

BIB = "references.bib"
src = open(BIB, encoding="utf-8").read()

# 1. drop comment-only lines
kept, dropped_comments = [], 0
for line in src.split("\n"):
    if line.lstrip().startswith("%"):
        dropped_comments += 1
        continue
    kept.append(line)
out = "\n".join(kept)

# 2. drop whole `note = {...}` fields (balanced single-level braces)
note_re = re.compile(r"[ \t]*\bnote\b\s*=\s*\{[^{}]*\}\s*,?[ \t]*\n", re.I)
notes = note_re.findall(out)
out = note_re.sub("", out)

# collapse any blank line left behind inside an entry
out = re.sub(r"\n{3,}", "\n\n", out)

print(f"comment lines dropped: {dropped_comments}")
print(f"note fields removed:   {len(notes)}")
for n in notes:
    print("   -", " ".join(n.split())[:100])

# sanity: entry count must not change
before = len(re.findall(r"^@", src, re.M))
after = len(re.findall(r"^@", out, re.M))
assert before == after, f"entry count changed: {before} -> {after}"
print(f"entries intact: {after}")

if "--write" in sys.argv:
    open(BIB, "w", encoding="utf-8").write(out)
    print("written")
else:
    print("(dry run; pass --write to apply)")
