"""Strip LaTeX comments from a .tex file without changing the typeset output.

Two things must survive, or the rendering silently breaks:

  * ``\\%``  is an escaped percent sign in the text (e.g. ``1\\%``), not a comment.
  * A bare ``%`` at end of line is a line-join that suppresses the newline's
    space. ``\\resizebox{\\textwidth}{!}{%`` relies on it; deleting it would
    inject a space into the box. We keep those and only remove comment *text*.

Comment-only lines are dropped entirely.

Run from paper/:  python _strip_comments.py manuscript.tex [--write]
"""
import re
import sys


def split_comment(line):
    """Return (code, comment_text) splitting at the first unescaped '%'."""
    i = 0
    while i < len(line):
        if line[i] == "\\":       # skip the escaped character
            i += 2
            continue
        if line[i] == "%":
            return line[:i], line[i + 1:]
        i += 1
    return line, None


def strip(text):
    out, dropped, trimmed = [], 0, 0
    for line in text.split("\n"):
        code, comment = split_comment(line)
        if comment is None:
            out.append(line)
            continue
        if code.strip() == "":
            dropped += 1          # comment-only line: drop it
            continue
        if comment.strip() == "":
            out.append(line)      # bare '%' line-join: keep verbatim
            continue
        out.append(code.rstrip() + "%")   # keep the join, drop the words
        trimmed += 1
    return "\n".join(out), dropped, trimmed


path = sys.argv[1]
src = open(path, encoding="utf-8").read()
new, dropped, trimmed = strip(src)
print(f"{path}: dropped {dropped} comment-only lines, trimmed {trimmed} trailing comments")

if "--write" in sys.argv:
    open(path, "w", encoding="utf-8").write(new)
    print("written")
else:
    print("(dry run; pass --write to apply)")
