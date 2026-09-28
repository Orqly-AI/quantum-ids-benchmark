"""Replace the generated table environments in manuscript.tex with the current _tab_*.tex files.

Each table is located by its label; the enclosing \\begin{table} ... \\end{table} is swapped for the
regenerated file, so a change to a generator (or to the result files) is propagated by running the
generator and then this script. Labels absent from the manuscript are skipped.

Run from paper/:  python _gen_tables5.py && python _gen_tables6.py && python _refresh_tables.py
"""
PAIRS = {"tab:induced": "_tab_induced.tex", "tab:hybridhw": "_tab_hybridhw.tex",
         "tab:hardware": "_tab_hardware.tex"}
s = open("manuscript.tex", encoding="utf-8").read()
for label, path in PAIRS.items():
    tag = "\\label{" + label + "}"
    if tag not in s:
        print("skip", label)
        continue
    i = s.index(tag)
    start = s.rfind("\\begin{table}", 0, i)
    end = s.index("\\end{table}", i) + len("\\end{table}")
    new = open(path, encoding="utf-8").read().strip()
    assert tag in new, path
    s = s[:start] + new + s[end:]
    print("refreshed", label, "from", path)
open("manuscript.tex", "w", encoding="utf-8").write(s)
