#!/usr/bin/env bash
# Build every artefact from manuscript.tex, in dependency order.
#
#   manuscript.pdf                          master (inline TikZ, elsarticle)
#   arxiv/    -> arXiv package, PNG figures        (author's choice)
#   submission/ -> journal package, vector PDF figures
#
# Run:  bash _build_all.sh
set -eu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
cd "$(dirname "$0")"

echo "== 0/6  generated tables and Figure 6 from the result files"
python _gen_tables4.py >/dev/null && python _gen_tables5.py >/dev/null && python _gen_tables6.py >/dev/null \
  && python _gen_tables7.py >/dev/null && python _gen_tables8.py >/dev/null && python _gen_tables9.py >/dev/null \
  && python _gen_fig6.py >/dev/null
python _refresh_tables.py | sed 's/^/   /'

echo "== 1/6  master manuscript"
rm -f manuscript.bbl manuscript.aux
tectonic manuscript.tex --keep-intermediates --reruns 5 >/tmp/b1.log 2>&1
grep -cE '^error:' /tmp/b1.log | sed 's/^/   errors: /'

echo "== 2/6  figures -> figs.pdf -> Fig1..6.pdf"
python _sync_figs.py
( cd arxiv && tectonic figs.tex >/tmp/b2.log 2>&1 && pdfseparate figs.pdf Fig%d.pdf )

echo "== 3/6  Fig1..6.png (600 dpi, for arXiv)"
bash _render_png.sh

echo "== 4/6  arXiv package (PNG figures)"
python _sync_arxiv.py --ext png --out arxiv/main.tex
cp references.bib arxiv/
( cd arxiv && rm -f main.bbl main.aux \
  && tectonic main.tex --keep-intermediates --reruns 5 >/tmp/b3.log 2>&1 \
  && grep -cE '^error:' /tmp/b3.log | sed 's/^/   errors: /' )
python - <<'PY'
import zipfile, os
os.chdir('arxiv')
files = ['main.tex', 'main.bbl', 'references.bib',
         'Fig1.png', 'Fig2.png', 'Fig3.png', 'Fig4.png', 'Fig5.png', 'Fig6.png']
with zipfile.ZipFile('arxiv_submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        assert os.path.exists(f), f
        z.write(f)
print("   arxiv/arxiv_submission.zip:", ", ".join(files))
PY

echo "== 5/6  journal package (vector PDF figures)"
mkdir -p submission
python _sync_arxiv.py --ext pdf --out submission/main.tex
cp references.bib submission/
for n in 1 2 3 4 5 6; do cp "arxiv/Fig${n}.pdf" submission/; done
( cd submission && rm -f main.bbl main.aux \
  && tectonic main.tex --keep-intermediates --reruns 5 >/tmp/b4.log 2>&1 \
  && grep -cE '^error:' /tmp/b4.log | sed 's/^/   errors: /' )
python - <<'PY'
import zipfile, os
files = ['main.tex', 'main.bbl', 'references.bib',
         'Fig1.pdf', 'Fig2.pdf', 'Fig3.pdf', 'Fig4.pdf', 'Fig5.pdf', 'Fig6.pdf']
with zipfile.ZipFile('submission_qip.zip', 'w',
                     zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = os.path.join('submission', f)
        assert os.path.exists(p), p
        z.write(p, f)
print("   submission_qip.zip:", ", ".join(files))
PY

echo "== 6/6  cover letter + form metadata"
tectonic cover_letter.tex >/tmp/b5.log 2>&1
python _submission_meta.py >/dev/null
echo "   cover_letter.pdf, _submission_metadata.txt"

echo
echo "page counts:"
for f in manuscript.pdf arxiv/main.pdf submission/main.pdf cover_letter.pdf; do
  printf '   %-28s %s\n' "$f" "$(pdfinfo "$f" | awk '/^Pages/{print $2}')"
done
