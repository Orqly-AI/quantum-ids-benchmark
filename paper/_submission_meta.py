"""Emit paste-ready plain-text title / abstract / keywords for the Elsevier
submission form, extracted from manuscript.tex so the form metadata cannot
drift from the uploaded PDF.

Run from paper/:  python _submission_meta.py
"""
import re

src = open("manuscript.tex", encoding="utf-8").read()


def block(env):
    m = re.search(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", src, re.DOTALL)
    assert m, f"no {env} block found"
    return m.group(1).strip()


# ---- title ----
m = re.search(r"\\title\{(.*?)\}\s*\n\s*\n", src, re.DOTALL)
assert m, "no title found"
title = " ".join(m.group(1).split())

# ---- abstract -> plain text ----
a = block("abstract")
a = re.sub(r"\\emph\{(.*?)\}", r"\1", a)
a = re.sub(r"\\textbf\{(.*?)\}", r"\1", a)


def _plain_math(m):
    """$...$ -> plain text: keep digits/letters, spell out the few macros used."""
    t = m.group(1)
    t = t.replace(r"\times", "x").replace(r"\%", "%").replace(r"\ge", ">=").replace(r"\le", "<=")
    t = t.replace(r"\rho", "rho").replace("{,}", ",")
    t = re.sub(r"(?<=[A-Za-z])=(?=[0-9])", " = ", t)
    return t


a = re.sub(r"\$([^$]*)\$", _plain_math, a)
a = a.replace(r"et al.\ ", "et al. ").replace(r"\ ", " ").replace(r"\%", "%").replace("~", " ").replace("{,}", ",")
a = " ".join(a.split())
leftover_math = re.findall(r"[\\${}].{0,20}", a)
assert not leftover_math, "LaTeX left in abstract: " + repr(leftover_math)

# ---- keywords ----
kws = [" ".join(x.split()) for x in block("keyword").split(r"\sep")]
kw_line = "; ".join(kws)

leftover = re.findall(r"[\\${}]", a)
report = (
    f"TITLE ({len(title)} chars)\n{title}\n\n"
    f"ABSTRACT ({len(a.split())} words, {len(a)} chars)\n{a}\n\n"
    f"KEYWORDS ({len(kws)}/10)\n{kw_line}\n"
)
open("_submission_metadata.txt", "w", encoding="utf-8").write(report)
print(report)
print("residual LaTeX markup:", leftover if leftover else "none")
