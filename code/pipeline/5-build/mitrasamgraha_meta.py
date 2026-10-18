#!/usr/bin/env python3
"""Metadata for the Mitrasamgraha sa-en texts, for the dict-webapp ingester.

Two sides per text:
  * Sanskrit side -> the DharmaNexus Sanskrit record (SA_files.json), so the
    Mitrasamgraha evidence is filed under the SAME text as the sa-bo / sa-zh
    evidence (displayName, author, collection/category, date estimate).
    Matching: exact filenr, then case-insensitive, then difflib fuzzy over filenr
    (>= 0.75), then title similarity; ambiguous / multi-file texts are pinned by
    OVERRIDES (reviewed by hand 2026-09-07). Texts with no single DharmaNexus
    file (Skanda-Purāṇa spread over 7 khaṇḍa files, Pāli Dhammapada, the mixed
    Itihāsa extracts) get a synthetic SA_MS_* record.
  * English side -> a synthetic EN_MS_* record carrying the Mitrasamgraha
    catalog data: title, translator, date, period, category, license.

Output: sa-en/mitrasamgraha_meta.json
  {"map": {<tsv stem>: {"sa_file", "en_file"}},
   "sa": {"files": [...synthetic SA records...], "category_names": [...]},
   "en": {"files": [...], "category_names": [...], "collection_names": [...]}}
The ingester's MetadataResolver merges the "sa" / "en" blocks into the
DharmaNexus tables via {SA,EN}_META_EXTRA.

  python mitrasamgraha_meta.py            # writes the json, prints the mapping table
"""
import csv, difflib, glob, json, os, re

DATA = "/home/sebastian/code/mitrasamgraha-dataset"
SA_META = os.path.expanduser("~/data/dharmanexus-sanskrit/SA_files.json")
OUT = "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-en/mitrasamgraha_meta.json"

# hand-reviewed pins: tsv stem -> DharmaNexus filename (None = synthetic record)
OVERRIDES = {
    "G14garsa-eng": "SA_GR14_garsa",
    "GE07bhg": "SA_GE07_bhg",                # plain mūla, not the commentary files
    "GE07mbh-ganguly": "SA_GE07_mbh_u",      # whole-epic umbrella record
    "GK19asvbc": "SA_GK19_asvbc_1u",         # Buddhacarita
    "GK19asvbh": "SA_GK19_asvbc_1u",         # Buddhacarita (Cowell), same text
    "GK19kaku": "SA_GK19_kakumspu",          # Kumārasambhava (fuzzy picked Gāhākoṣa)
    "GK19meghd": "SA_GK19_kmeghdpu",         # plain Meghadūta, not the Mallinātha file
    "GK19ragh": "SA_GK19_kragh_pu",          # plain Raghuvaṃśa, not the Vallabhadeva file
    "GK21sokss": "SA_GK21_sokss_pu",
    "GP12skp": None,                         # 7 khaṇḍa files in DharmaNexus -> synthetic umbrella
    "GP12visnup": "SA_GP12_visnup_u",
    "GV01rvh": "SA_GV01_rvh",
    "GVKyajurvedu": "SA_GV_kyajurvedu",      # case: fuzzy picked the Śukla file
    "Kv": "SA_K10_arthaviniscaya",
    "Ppdhp": None,                           # Pāli Dhammapada: not in the Sanskrit repo
    "T06sthmavt_u": "SA_T06_sthmavt",
    "T07vakobhku": "SA_T07_vakobhk",
    "itihasa-extract": None,                 # Rāmāyaṇa + Mahābhārata extracts (Itihāsa dataset)
}

# texts that are in the data but not in final_catalog.tsv (val/test-only texts)
EXTRA_CATALOG = {
    "GE07bhg": dict(Title="Bhagavadgītā", Date="400BCE-400CE", Translator="A. C. Bhaktivedanta Swami Prabhupāda",
                    period="Epic", category="rel_script", License="", note="Bhagavad-gītā As It Is (1972)"),
    "GP10bhp": dict(Title="Bhāgavata-Purāṇa", Date="800-1000", Translator="A. C. Bhaktivedanta Swami Prabhupāda",
                    period="Classical", category="purana", License="", note="Śrīmad-Bhāgavatam"),
    "K10vimkn_u": dict(Title="Vimalakīrtinirdeśa", Date="100-300", Translator="Robert A. F. Thurman",
                       period="Classical", category="rel_script", License="", note="The Holy Teaching of Vimalakīrti (1976)"),
    "GK19asvbc": dict(Title="Buddhacarita", Date="100-200", Translator="E. B. Cowell",
                      period="Classical", category="poetry", License="cc0", note="Sacred Books of the East 49 (1894)"),
    "GK19asvbh": dict(Title="Buddhacarita", Date="100-200", Translator="E. B. Cowell",
                      period="Classical", category="poetry", License="cc0", note="Sacred Books of the East 49 (1894)"),
    "itihasa-extract": dict(Title="Itihāsa extracts (Rāmāyaṇa, Mahābhārata)", Date="400BCE-400CE",
                            Translator="M. N. Dutt; K. M. Ganguli", period="Epic", category="epic", License="cc0",
                            note="Itihāsa dataset (Aralikatte et al. 2021)"),
}

CATEGORY_NAMES = {"rel_script": "Religious scripture", "epic": "Epic", "poetry": "Poetry",
                  "purana": "Purāṇa", "treatises": "Treatises", "vedic": "Vedic"}
SA_SYNTH = {   # synthetic Sanskrit records for texts without one DharmaNexus file
    "GP12skp": dict(displayName="Skanda-Purāṇa", collection="Purana", category="GP12", author=""),
    "Ppdhp": dict(displayName="Dhammapada (Pāli)", collection="Buddhist_Scriptures", category="MS_pali", author=""),
    "itihasa-extract": dict(displayName="Itihāsa extracts (Rāmāyaṇa, Mahābhārata)", collection="Epic",
                            category="MS_itihasa", author=""),
}
SA_EXTRA_CATEGORIES = [{"category": "MS_pali", "displayName": "Pāli scriptures"},
                       {"category": "MS_itihasa", "displayName": "Itihāsa (Rāmāyaṇa & Mahābhārata extracts)"}]


def stem_of(fname):
    s = re.sub(r"\.tsv$", "", os.path.basename(fname))
    s = re.sub(r"_(train|val|test)$", "", s)
    s = re.sub(r"^itihasa-(val|test)-extract$", "itihasa-extract", s)
    return s


def fuzzy_match(stem, title, by_filenr, records):
    cands = [stem, stem.split("-")[0]]
    for c in cands:
        if c in by_filenr:
            return by_filenr[c], "exact"
    lower = {k.lower(): v for k, v in by_filenr.items()}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()], "case"
    m = difflib.get_close_matches(stem, list(by_filenr), n=1, cutoff=0.75)
    if m:
        return by_filenr[m[0]], f"fuzzy-filenr({m[0]})"
    if title:
        titles = {(r.get("uniformtitle") or r.get("displayName") or ""): r for r in records}
        m = difflib.get_close_matches(title, list(titles), n=1, cutoff=0.8)
        if m:
            return titles[m[0]], f"fuzzy-title({m[0]})"
    return None, "none"


def main():
    records = json.load(open(SA_META, encoding="utf-8"))
    by_name = {r["filename"]: r for r in records}
    by_filenr = {}
    for r in records:
        by_filenr.setdefault(r.get("filenr") or "", r)
    catalog = {}
    for r in csv.DictReader(open(f"{DATA}/final_catalog.tsv", encoding="utf-8"), delimiter="\t"):
        catalog[stem_of(r["Filename"])] = r
    catalog.update({k: v for k, v in EXTRA_CATALOG.items() if k not in catalog})

    stems = sorted({stem_of(p) for p in glob.glob(f"{DATA}/data/**/*.tsv", recursive=True)})
    mapping, sa_files, en_files = {}, [], []
    print(f"{'stem':18} {'how':22} {'sa_file':24} {'SA displayName':42} | EN title / translator")
    for s in stems:
        cat = catalog.get(s, {})
        if s in OVERRIDES:
            sa_rec, how = (by_name[OVERRIDES[s]], "override") if OVERRIDES[s] else (None, "synthetic")
        else:
            sa_rec, how = fuzzy_match(s, cat.get("Title"), by_filenr, records)
        if sa_rec is None:
            synth = SA_SYNTH.get(s) or dict(displayName=cat.get("Title", s), collection="Other", category="?", author="")
            sa_file = f"SA_MS_{s}"
            sa_files.append({"filename": sa_file, "textname": sa_file, "filenr": sa_file, **synth,
                             "date": cat.get("Date", ""), "period": cat.get("period", "")})
            sa_disp = synth["displayName"]
        else:
            sa_file, sa_disp = sa_rec["filename"], sa_rec["displayName"]
        en_file = f"EN_MS_{s}"
        title = cat.get("Title") or (sa_rec or {}).get("uniformtitle") or sa_disp
        en_files.append({
            "filename": en_file, "textname": en_file, "filenr": en_file,
            "displayName": title.strip(), "translator": [cat.get("Translator", "").strip()] if cat.get("Translator") else [],
            "collection": "Mitrasamgraha", "category": cat.get("category", "?"),
            "date": cat.get("Date", "").strip(), "period": cat.get("period", "").strip(),
            "license": cat.get("License", "").strip(), "note": cat.get("note", ""),
            "sa_file": sa_file, "source": "Mitrasaṃgraha (Nehrdich et al.)",
        })
        mapping[s] = {"sa_file": sa_file, "en_file": en_file}
        print(f"{s:18} {how:22} {sa_file:24} {sa_disp[:42]:42} | {title[:30]} / {cat.get('Translator','?')}")

    out = {"map": mapping,
           "sa": {"files": sa_files, "category_names": SA_EXTRA_CATEGORIES},
           "en": {"files": en_files,
                  "category_names": [{"category": k, "displayName": v} for k, v in CATEGORY_NAMES.items()]
                                    + [{"category": "?", "displayName": "Unclassified"}],
                  "collection_names": [{"collection": "Mitrasamgraha",
                                        "displayName": "Mitrasaṃgraha (Sanskrit–English translations)"}]}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{len(mapping)} texts, {len(sa_files)} synthetic SA records, {len(en_files)} EN records -> {OUT}")


if __name__ == "__main__":
    main()
