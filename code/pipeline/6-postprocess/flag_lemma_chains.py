#!/usr/bin/env python3
"""Apply the e-text error test of postprocess_final.py to lemma chains (src_lemma) of the
webapp's *_combined.ndjson rows.  Input: TSV <src_lemma>\t<rows>; output: TSV of flagged chains
<src_lemma>\t<rows>\t<member>\t<suspected_form>.  A chain is flagged if one of its members
(split at '-', ' ', '_', >= 3 chars) passes the test (rare in dharmanexus, one edit from a
frequent lemma/form, not an MW/Apte headword)."""
import importlib.util, os, re, sys
spec = importlib.util.spec_from_file_location("pp", os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprocess_final.py"))
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)

def test(chain):
    for m in (x for x in re.split(r"[- _]+", pp.norm(chain)) if len(x) >= 3):
        if pp.cnt(m) <= pp.RARE and pp.ref_key(m) not in pp.ALLOW:
            nb = pp.neighbour(m)
            if nb:
                return m, nb
    return None

for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    n = rows = fr = 0
    with open(dst, "w") as o:
        for l in open(src):
            k, c = l.rstrip("\n").rsplit("\t", 1); c = int(c); n += 1; rows += c
            r = test(k) if k else None
            if r:
                o.write(f"{k}\t{c}\t{r[0]}\t{r[1]}\n"); fr += c
    print(os.path.basename(src), "chains", n, "rows", rows, "flagged chains", sum(1 for _ in open(dst)), "flagged rows", fr, f"({100*fr/rows:.2f}%)")
