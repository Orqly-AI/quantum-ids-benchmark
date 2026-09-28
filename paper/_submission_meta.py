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
a = a.replace(r"$0.1\%/1\%$", "0.1%/1%")
a = a.replace(r"$8$ to $14\times$", "8 to 14x")
a = a.replace(r"$1\%$", "1%")
a = a.replace(r"$p=0.005$", "p = 0.005")
a = a.replace(r"$q=0.030$", "q = 0.030")
a = a.replace(r"\%", "%")
a = " ".join(a.split())

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
