"""Write _arxiv_metadata.txt (paste-ready arXiv form fields) from the built artefacts.

Title, abstract and keywords come from _submission_metadata.txt (written by
_submission_meta.py from manuscript.tex); the page count from arxiv/main.pdf;
figure and table counts from manuscript.tex; the upload listing from the zip.

Run from paper/ after _build_all.sh:  python _arxiv_meta.py
"""
import re
import subprocess
import zipfile

meta = open("_submission_metadata.txt", encoding="utf-8").read()
title = meta.split("TITLE")[1].split("\n", 1)[1].split("\n\n")[0].strip()
abstract = meta.split("ABSTRACT")[1].split("\n", 1)[1].split("\n\n")[0].strip()
src = open("manuscript.tex", encoding="utf-8").read()
n_fig = len(re.findall(r"\\begin\{figure\}", src))
n_tab = len(re.findall(r"\\begin\{table\}", src))
pages = subprocess.run(["pdfinfo", "arxiv/main.pdf"], capture_output=True, text=True).stdout
n_pages = int(re.search(r"Pages:\s+(\d+)", pages).group(1))
files = zipfile.ZipFile("arxiv/arxiv_submission.zip").namelist()

# wrap the abstract at 80 columns for readability; arXiv ignores line breaks
words, lines, cur = abstract.split(), [], ""
for w in words:
    if len(cur) + len(w) + 1 > 80:
        lines.append(cur)
        cur = w
    else:
        cur = (cur + " " + w).strip()
lines.append(cur)

out = f"""arXiv SUBMISSION METADATA (replacement version of 2608.18155)
=============================================================

UPLOAD FILE
  paper/arxiv/arxiv_submission.zip
  ({', '.join(files)})

LICENSE
  arXiv.org perpetual, non-exclusive license (as for v1)

PRIMARY CATEGORY
  quant-ph   (Quantum Physics)

CROSS-LIST
  cs.LG      (Machine Learning)
  cs.CR      (Cryptography and Security)

TITLE
{title}

AUTHORS (arXiv comma-separated form, submission order; unchanged from v1)
Syeda Anshrah Gillani, Mirza Samad Ahmed Baig, Shahid Munir Shah, Asher Ali, Hamzah Siddiqui

COMMENTS
{n_pages} pages, {n_fig} figures, {n_tab} tables. v2: adds a kernel-level analysis of the
quantum-kernel result, an induced-shift experiment on UNSW-NB15, and runs of
the projected kernel and the 4-qubit hybrid on three IBM Heron processors.
Code, configurations, fixed seeds, per-run result records and raw device
measurements: https://github.com/Orqly-AI/quantum-ids-benchmark

JOURNAL REF / DOI / REPORT NUMBER
  leave all blank (under review)

ACM CLASS (optional)
  I.2.6; C.2.0

ABSTRACT ({len(words)} words; plain text, no LaTeX; paste as-is)
--------------------------------------------------------
""" + "\n".join(lines) + "\n"
open("_arxiv_metadata.txt", "w", encoding="utf-8").write(out)
print(f"wrote _arxiv_metadata.txt: {n_pages} pages, {n_fig} figures, {n_tab} tables, "
      f"{len(words)}-word abstract, {len(files)} files in zip")
