"""Composition of the released dictionaries by the source of their attestations (2026-10-11).

Counts the attestations of the Sanskrit->X dictionaries (every attestation, since the dictionaries carry all of
them) and the distinct sentence pairs they come from:
  sa-en:        genre x period of the Mitrasamgraha text (final_catalog.tsv, as saen_genre.py)
  sa-bo, sa-zh: DharmaNexus collection and category of the Sanskrit text (SA_files.json, SA_category-names.json),
                read from the sentence id (SA_<category>_<text>:<n>_...)
-> notes/eval/attestation-composition.json
"""
import json, os, re, collections

MINED = os.environ.get("EVAL_MINED_DIR", "/mnt2/code/mitra-dictionaries/paper-final-output-20261011")
DNX = os.path.expanduser("~/data/dharmanexus-sanskrit")
OUT = "/home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/attestation-composition.json"
SPLIT = re.compile(r"_(train|test|val|valid|validation|dev)(?=\.tsv$)")


def ids(f):
    """Yield the sentence id of every attestation of a sa2X dictionary."""
    rx = re.compile(r'"id": "([^"]+)"')
    for line in open(f, encoding="utf-8"):
        yield from rx.findall(line)


def saen():
    import csv
    cat = {}
    for r in csv.DictReader(open(os.path.expanduser("~/code/mitrasamgraha-dataset/final_catalog.tsv"), encoding="utf-8"), delimiter="\t"):
        cat[SPLIT.sub("", r["Filename"])] = (r["category"], r["period"])
    extra = {"K10vimkn_u.tsv": ("rel_script", "Classical"), "GP10bhp.tsv": ("purana", "Classical"),   # = saen_genre.EXTRA
             "GK19asvbh.tsv": ("poetry", "Classical"), "GK19asvbc.tsv": ("poetry", "Classical"),
             "GE07bhg.tsv": ("rel_script", "Epic")}
    att, sents, miss = collections.Counter(), collections.defaultdict(set), collections.Counter()
    for sid in ids(f"{MINED}/skt_en_bidict.sa2en.ndjson"):
        fn = SPLIT.sub("", sid.split(":")[0].split("/")[-1])
        c = cat.get(fn) or (("epic", "Epic") if fn.startswith("itihasa") else None) or extra.get(fn)
        if not c:
            miss[fn] += 1; continue
        att[c] += 1; sents[c].add(sid)
    return {"by_genre_period": [dict(genre=g, period=p, attestations=v, sentence_pairs=len(sents[(g, p)]))
                                for (g, p), v in sorted(att.items())],
            "attestations": sum(att.values()), "sentence_pairs": sum(len(s) for s in sents.values()),
            "unmapped": sum(miss.values()), "unmapped_files": miss.most_common(10)}


def dnx(pfx, tag):
    files = json.load(open(f"{DNX}/SA_files.json"))
    coll = {f["category"]: f["collection"] for f in files}
    cname = {c["category"]: c["displayName"] for c in json.load(open(f"{DNX}/SA_category-names.json"))}
    att, sents, texts = collections.Counter(), collections.defaultdict(set), collections.defaultdict(set)
    for sid in ids(f"{MINED}/{pfx}_bidict.sa2{tag}.ndjson"):
        m = re.match(r"SA_([A-Z]+\d+)_([^:]+):", sid)
        c = m.group(1) if m else "?"
        att[c] += 1; sents[c].add(sid.split("_" + tag.upper() + "_")[0]); texts[c].add(m.group(2) if m else "?")
    rows = [dict(category=c, name=cname.get(c, c), collection=coll.get(c, "?"), attestations=v,
                 sanskrit_sentences=len(sents[c]), texts=len(texts[c])) for c, v in att.most_common()]
    bycoll = collections.Counter()
    for r in rows: bycoll[r["collection"]] += r["attestations"]
    return {"by_category": rows, "by_collection": dict(bycoll), "attestations": sum(att.values())}


if __name__ == "__main__":
    res = {"mined_dir": MINED, "sa-en": saen(), "sa-bo": dnx("skt_tib", "bo"), "sa-zh": dnx("skt_zh", "zh")}
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
