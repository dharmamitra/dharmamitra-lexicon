import csv, json, os, re, collections
cat = {}
for r in csv.DictReader(open(os.path.expanduser("~/code/mitrasamgraha-dataset/final_catalog.tsv"), encoding="utf-8"), delimiter="\t"):
    stem = re.sub(r"_(train|test|val|valid|validation|dev)(?=\.tsv$)", "", r["Filename"])
    cat[stem] = (r["category"], r["period"])
# held-out files of Mitrasamgraha that are not in final_catalog.tsv (genre assigned by us)
EXTRA = {"K10vimkn_u.tsv": ("rel_script", "Classical"), "GP10bhp.tsv": ("purana", "Classical"),
         "GK19asvbh.tsv": ("poetry", "Classical"), "GK19asvbc.tsv": ("poetry", "Classical"),
         "GE07bhg.tsv": ("rel_script", "Epic")}
att = collections.Counter(); sents = collections.defaultdict(set); miss = collections.Counter()
F = "/mnt2/code/mitra-dictionaries/qwen-dictionaries/final-output/pre-mdc-20260919/skt_en_spans.ndjson"
for line in open(F, encoding="utf-8"):
    i = line.find('"sentence_id": "'); sid = line[i + 16:line.find('"', i + 16)]
    fn = re.sub(r"_(train|test|val|valid|validation|dev)(?=\.tsv$)", "", sid.split(":")[0].split("/")[-1])
    c = cat.get(fn) or (("epic", "Epic") if fn.startswith("itihasa") else None) or EXTRA.get(fn)
    if not c:
        miss[fn] += 1; continue
    att[c[0]] += 1; sents[c[0]].add(sid)
tot = sum(att.values()); ts = sum(len(v) for v in sents.values())
print("attestations", tot, "unmapped", sum(miss.values()), miss.most_common(40))
for k, v in att.most_common():
    print(f"{k:12} attestations {v:8d} {100*v/tot:5.1f}%   sentence pairs {len(sents[k]):7d} {100*len(sents[k])/ts:5.1f}%")
json.dump({k: dict(attestations=v, sentence_pairs=len(sents[k])) for k, v in att.items()}, open("/home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/saen-genre.json", "w"), indent=1)
