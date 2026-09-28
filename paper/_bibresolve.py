"""Resolve placeholder author fields in references.bib from public metadata APIs.

Entries carrying a DOI are looked up via Crossref; entries carrying an arXiv
eprint id via the arXiv API. Prints the real author list for each so it can be
checked before patching. Use --write to apply.

Run from paper/:  python _bibresolve.py [--write]
"""
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

BIB = "references.bib"
UA = {"User-Agent": "bib-resolver/1.0 (academic reference check)"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def from_crossref(doi):
    d = json.loads(get(f"https://api.crossref.org/works/{doi}"))["message"]
    names = []
    for a in d.get("author", []):
        fam, given = a.get("family"), a.get("given")
        names.append(f"{fam}, {given}" if fam and given else (fam or given or ""))
    return " and ".join(n for n in names if n), d.get("container-title", [""])[0]


def from_arxiv(eid):
    # the arXiv API rate-limits hard; pace requests and retry once
    xml = None
    for attempt in range(3):
        time.sleep(4 if attempt == 0 else 12)
        try:
            xml = get(f"http://export.arxiv.org/api/query?id_list={eid}")
            break
        except Exception:
            if attempt == 2:
                raise
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = ET.fromstring(xml).find("a:entry", ns)
    names = []
    for a in entry.findall("a:author", ns):
        full = a.find("a:name", ns).text.strip()
        parts = full.split()
        names.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else full)
    return " and ".join(names), "arXiv preprint"


bib = open(BIB, encoding="utf-8").read()
blocks = list(re.finditer(r"^@([a-z]+)\{([^,]+),(.*?)(?=^@|\Z)", bib, re.M | re.S))
resolved, failed = {}, []

for m in blocks:
    key, body = m.group(2).strip(), m.group(3)
    if "Anonymous" not in (re.search(r"author\s*=\s*\{(.*?)\}\s*,", body, re.S) or
                           type("x", (), {"group": lambda s, n: ""})()).group(1):
        continue
    doi = re.search(r"doi\s*=\s*\{([^}]+)\}", body)
    eid = re.search(r"eprint\s*=\s*\{([^}]+)\}", body)
    try:
        if doi:
            authors, venue = from_crossref(doi.group(1).strip())
            src = f"Crossref {doi.group(1).strip()}"
        elif eid:
            authors, venue = from_arxiv(eid.group(1).strip())
            src = f"arXiv {eid.group(1).strip()}"
        else:
            failed.append((key, "no DOI and no arXiv id"))
            continue
        if not authors:
            failed.append((key, f"{src} returned no authors"))
            continue
        resolved[key] = authors
        print(f"OK   {key}\n     via {src}\n     {authors}\n     venue: {venue}\n")
    except Exception as e:
        failed.append((key, f"{type(e).__name__}: {e}"))

for key, why in failed:
    print(f"FAIL {key}: {why}")

if "--write" in sys.argv and resolved:
    out = bib
    for key, authors in resolved.items():
        pat = re.compile(r"(@[a-z]+\{" + re.escape(key) + r",.*?author\s*=\s*)\{\{Anonymous\}\}",
                         re.S)
        out, n = pat.subn(lambda mm: mm.group(1) + "{" + authors + "}", out, count=1)
        assert n == 1, f"could not patch {key}"
    open(BIB, "w", encoding="utf-8").write(out)
    print(f"\npatched {len(resolved)} entries in {BIB}")
else:
    print(f"\n(dry run: {len(resolved)} resolvable, {len(failed)} not; pass --write to apply)")
