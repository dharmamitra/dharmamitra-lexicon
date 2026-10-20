#!/usr/bin/env python3
"""Domain distribution of the textual evidence behind Negi vs. the Sanskrit-Tibetan sentence pairs of MITRA-dict.

Negi: citations of texts (non-lexical sigla) with their locus, grouped by a hand-made siglum -> text -> domain map
(identifications are ours; sigla we could not identify go to 'unidentified').
Ours: sentence pairs of sa-bo/hybrid/final.ndjson by the DharmaNexus category of the Sanskrit text, grouped into
the same domains. Output: /home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/negi-domains.json and a printed table.
"""
import os, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import negi

FINAL = __import__("os").environ.get("CORPUS_SA_BO", "/mnt2/code/mitra-dictionaries/qwen-dictionaries/sa-bo/hybrid/final.ndjson")
DOMAINS = ["Sūtra", "Prajñāpāramitā", "Avadāna, Jātaka", "Vinaya", "Abhidharma", "Madhyamaka, Yogācāra",
           "Pramāṇa", "Tantra", "Kāvya, other", "unidentified"]
# siglum -> (text, domain); identifications ours, '?' marks a guess
NEGI_MAP = {
    "अ.क.": ("Avadānakalpalatā", "Avadāna, Jātaka"), "जा.मा.": ("Jātakamālā", "Avadāna, Jātaka"),
    "अ.श.": ("Avadānaśataka", "Avadāna, Jātaka"),
    "त.प.": ("Tattvasaṅgrahapañjikā", "Pramāṇa"), "त.स.": ("Tattvasaṅgraha", "Pramāṇa"),
    "प्र.अ.": ("Pramāṇavārttikālaṅkāra", "Pramāṇa"), "न्या.टी.": ("Nyāyabinduṭīkā", "Pramāṇa"),
    "प्र.वा.": ("Pramāṇavārttika", "Pramāṇa"), "वा.टी.": ("Vādanyāyaṭīkā", "Pramāṇa"),
    "वि.प्र.": ("Vimalaprabhā", "Tantra"), "हे.त.": ("Hevajratantra", "Tantra"), "म.मू.": ("Mañjuśrīmūlakalpa", "Tantra"),
    "स.दु.": ("Sarvadurgatipariśodhana", "Tantra"), "गु.स.": ("Guhyasamāja", "Tantra"), "गु.सि.": ("Guhyasiddhi", "Tantra"),
    "वि.सू.": ("Vinayasūtra", "Vinaya"),
    "अभि.भा.": ("Abhidharmakośabhāṣya", "Abhidharma"), "अभि.स्फु.": ("Sphuṭārthā", "Abhidharma"),
    "अभि.स.भा.": ("Abhidharmasamuccayabhāṣya", "Abhidharma"), "अभि.को.": ("Abhidharmakośa", "Abhidharma"),
    "ल.अ.": ("Laṅkāvatāra", "Sūtra"), "ग.व्यू.": ("Gaṇḍavyūha", "Sūtra"), "ल.वि.": ("Lalitavistara", "Sūtra"),
    "द.भू.": ("Daśabhūmika", "Sūtra"), "स.पु.": ("Saddharmapuṇḍarīka", "Sūtra"), "का.व्यू.": ("Kāraṇḍavyūha", "Sūtra"),
    "रा.प.": ("Rāṣṭrapālaparipṛcchā", "Sūtra"), "सु.प्र.": ("Suvarṇaprabhāsa", "Sūtra"),
    "अ.सा.": ("Aṣṭasāhasrikā", "Prajñāpāramitā"),
    "बो.भू.": ("Bodhisattvabhūmi", "Madhyamaka, Yogācāra"), "शि.स.": ("Śikṣāsamuccaya", "Madhyamaka, Yogācāra"),
    "बो.अ.": ("Bodhicaryāvatāra", "Madhyamaka, Yogācāra"), "सू.अ.": ("Mahāyānasūtrālaṅkāra", "Madhyamaka, Yogācāra"),
    "सू.व्या.": ("Sūtrālaṅkāra commentary", "Madhyamaka, Yogācāra"), "श्रा.भू.": ("Śrāvakabhūmi", "Madhyamaka, Yogācāra"),
    "र.वि.": ("Ratnagotravibhāga", "Madhyamaka, Yogācāra"), "म.भा.": ("Madhyāntavibhāgabhāṣya", "Madhyamaka, Yogācāra"),
    "म.टी.": ("Madhyāntavibhāgaṭīkā", "Madhyamaka, Yogācāra"), "त्रि.भा.": ("Triṃśikābhāṣya", "Madhyamaka, Yogācāra"),
    "प्र.प.": ("Prasannapadā", "Madhyamaka, Yogācāra"),
    "का.आ.": ("Kāvyādarśa", "Kāvya, other"), "मे.दू.": ("Meghadūta", "Kāvya, other"), "यो.श.": ("Yogaśataka", "Kāvya, other"),
}
# DharmaNexus Sanskrit categories -> domain
DN_MAP = {"K01": "Vinaya", "T08": "Vinaya", "K02": "Prajñāpāramitā", "K03": "Prajñāpāramitā", "K05": "Prajñāpāramitā",
          "K06": "Prajñāpāramitā", "K07": "Prajñāpāramitā", "T03": "Prajñāpāramitā", "K08": "Sūtra", "K09": "Sūtra",
          "K10": "Sūtra", "K14": "Sūtra", "T05": "Sūtra", "K12": "Tantra", "T02": "Tantra", "T07": "Abhidharma",
          "T04": "Madhyamaka, Yogācāra", "T06": "Madhyamaka, Yogācāra", "T11": "Pramāṇa", "T09": "Avadāna, Jātaka"}
SIG = re.compile(r"((?:[ऀ-ॿ]{1,7}\.\s?){1,4})\s?(\d[\dकखगघ/.\-]*)")
NOISE = {"सं.", "पृ.", "क्रि.", "ना.", "वि.", "पा.", "अ.", "प्र.", "को."}


def main():
    cites, loci, texts = collections.Counter(), collections.defaultdict(set), collections.defaultdict(set)
    for line in open(negi.NEGI, encoding="utf-8"):
        if "\t" not in line:
            continue
        for m in SIG.finditer(line.split("\t", 1)[1]):
            sg = re.sub(r"\s", "", m.group(1))
            if negi.is_lexical_siglum(sg) or sg in NOISE or sg[0] in "िाीुूेैोौ्ंः":
                continue
            text, dom = NEGI_MAP.get(sg, (sg, "unidentified"))
            cites[dom] += 1
            loci[dom].add((sg, m.group(2).rstrip(".")))
            texts[dom].add(text)
    ours, our_texts = collections.Counter(), collections.defaultdict(set)
    for line in open(FINAL, encoding="utf-8"):
        i = line.find('"src_segnr": ["SA_')
        cat = line[i + 18:i + 21] if i >= 0 else ""
        t = line[i + 15:line.find(":", i + 15)] if i >= 0 else ""
        dom = DN_MAP.get(cat, "Kāvya, other")
        if re.search(r"avsata|divyav|divav|avadan|jatak|jm_", t, re.I):   # avadāna collections filed under K10
            dom = "Avadāna, Jātaka"
        ours[dom] += 1
        our_texts[dom].add(t)
    nc, nl, no = sum(cites.values()), sum(len(v) for v in loci.values()), sum(ours.values())
    print(f"Negi: {nc:,} citations of texts, {nl:,} distinct loci, {sum(len(v) for v in texts.values())} sources | "
          f"MITRA-dict sa-bo: {no:,} sentence pairs, {sum(len(v) for v in our_texts.values())} Sanskrit texts")
    print(f"{'domain':24} {'Negi loci':>10} {'%':>6} {'sources':>8} | {'our pairs':>10} {'%':>6} {'texts':>6}")
    res = {}
    for d in DOMAINS:
        l, o = len(loci[d]), ours[d]
        res[d] = dict(negi_citations=cites[d], negi_loci=l, negi_sources=len(texts[d]), ours_pairs=o, ours_texts=len(our_texts[d]))
        print(f"{d:24} {l:10,d} {100*l/nl:6.1f} {len(texts[d]):8d} | {o:10,d} {100*o/no:6.1f} {len(our_texts[d]):6d}")
    json.dump(res, open("/home/sebastian/papers/mitra-dict-2026/notes/eval-mw99-20261020/negi-domains.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
