#!/usr/bin/env python3
"""Derive a word-segmented Chinese string from the kokuyaku Japanese (`par_japanese`).

Kundoku (訓読) renders classical Chinese with kana inserted as reading aids
(okurigana, particles, inflections) between the original Chinese morphemes, and
reorders to Japanese (SOV) syntax. So: drop the kana, turn each kana run into a
single space, and the kanji come out segmented at word boundaries -- in Japanese
(kundoku) word order. Character forms are then normalized from Japanese
shinjitai to Traditional Chinese (OpenCC jp2t) plus a small supplement for the
high-frequency residuals OpenCC misses (verified against the source: ZH uses
為 not 爲, 眾 not 衆).

Adds `par_zh_segmented` right after `par_japanese`. Resumable; atomic per-file.
"""
import argparse, json, os, unicodedata, opencc

# OpenCC jp2t leaves these two JP forms; the Taisho source uses the right-hand forms.
SUPPLEMENT = {"爲": "為", "衆": "眾"}
_CC = opencc.OpenCC("jp2t")


def is_kanji(ch):
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF or ch in "々〆〇"


def to_segmented_zh(japanese):
    """kana-strip -> single-space between kanji runs -> jp2t -> supplement."""
    text = unicodedata.normalize("NFC", japanese)
    out = []
    for ch in text:
        if is_kanji(ch):
            out.append(ch)
        elif out and out[-1] != " ":
            out.append(" ")
    s = " ".join("".join(out).split())
    s = _CC.convert(s)
    return "".join(SUPPLEMENT.get(c, c) for c in s)


def insert_after(d, anchor, key, value):
    out = {}
    for k, v in d.items():
        out[k] = v
        if k == anchor:
            out[key] = value
    if anchor not in d:
        out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()
    for fpath in args.files:
        records = []
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        n = added = 0
        for r in records:
            ja = (r.get("par_japanese") or "").strip()
            if not ja or r.get("par_zh_segmented") is not None:
                continue
            r["par_zh_segmented"] = to_segmented_zh(ja)
            added += 1
        if not added:
            print(f"{os.path.basename(fpath)}: nothing to do", flush=True)
            continue
        out_records = [
            insert_after(r, "par_japanese", "par_zh_segmented", r.pop("par_zh_segmented"))
            if "par_zh_segmented" in r else r
            for r in records
        ]
        tmp = fpath + ".tmp"
        with open(tmp, "w") as fh:
            for r in out_records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, fpath)
        print(f"{os.path.basename(fpath)}: added par_zh_segmented to {added} records", flush=True)


if __name__ == "__main__":
    main()
