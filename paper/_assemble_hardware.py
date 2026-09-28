"""Insert the hardware-validation material into manuscript.tex.

Reads:
  _sec_hardware_methods.tex     methods paragraph  -> after the 'Finite shots' paragraph (Sec. 4.6)
  _sec_hardware_results.tex     results subsection -> before 'Summary' in Section 7
  _tab_hardware.tex             table              -> inside that subsection (at @@TAB_HARDWARE@@)
  _sec_hardware_limitation.tex  replacement for the 'Simulation only' limitation paragraph

Idempotent: refuses to run twice.
"""
rd = lambda p: open(p, encoding="utf-8").read().strip()
s = open("manuscript.tex", encoding="utf-8").read()
assert r"\label{sec:khw}" not in s, "hardware section already inserted"

# 1. methods paragraph
anchor = "and the SVM is retrained on the resulting noisy features."
assert s.count(anchor) == 1, "finite-shots anchor"
s = s.replace(anchor, anchor + "\n\n" + rd("_sec_hardware_methods.tex"))

# 2. results subsection with table
res = rd("_sec_hardware_results.tex")
assert res.count("@@TAB_HARDWARE@@") == 1
res = res.replace("@@TAB_HARDWARE@@", rd("_tab_hardware.tex"))
anchor = r"\subsection{Summary}\label{sec:ksum}"
assert s.count(anchor) == 1, "summary anchor"
s = s.replace(anchor, res + "\n\n" + anchor)

# 3. limitations paragraph
start = s.index(r"\textbf{Simulation only.}")
end = s.index(r"\textbf{Qubit budget.}", start)
s = s[:start] + rd("_sec_hardware_limitation.tex") + "\n" + s[end:]

open("manuscript.tex", "w", encoding="utf-8").write(s)
print("hardware methods, results subsection, table and limitation inserted")
