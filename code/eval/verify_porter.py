#!/usr/bin/env python3
"""Verification helper (optional; needs an interpreter WITH nltk, e.g. ~/miniconda3/envs/mitra/bin/python):
compares english.porter() with nltk PorterStemmer(mode='ORIGINAL_ALGORITHM') on the vocabulary of all mined
sa->en glosses of single-word headwords + the MW72/Apte entry texts. Prints the number of disagreements."""
import os, sys, json, sqlite3, zlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, english as E

def vocab_worker(chunk):
    v = set()
    for line in C.iter_lines(*chunk):
        d = json.loads(line)
        if C.is_single_word(d["sa"]):
            for s in d["senses"]:
                v.update(E.tokens(s["en"]))
    return v

if __name__ == "__main__":
    from nltk.stem import PorterStemmer
    ps = PorterStemmer(mode="ORIGINAL_ALGORITHM")
    vocab = set()
    for v in C.parallel_scan(os.path.join(C.MINED_DIR, "skt_en_bidict.sa2en.ndjson"), vocab_worker):
        vocab |= v
    con = sqlite3.connect("file:/mnt2/code/mitra-dictionaries/alignment-annotator/app/dict.sqlite?mode=ro", uri=True)
    for (h,) in con.execute("select html from entries where dict in ('mw72','ap90')"):
        vocab.update(E.tokens(re.sub(r"<[^>]+>", " ", zlib.decompress(h).decode("utf-8"))))
    bad = [(w, E.porter(w), ps.stem(w, to_lowercase=False)) for w in sorted(vocab) if E.porter(w) != ps.stem(w, to_lowercase=False)]
    print(f"vocabulary: {len(vocab)} tokens; disagreements with nltk ORIGINAL_ALGORITHM: {len(bad)}")
    print(bad[:40])
