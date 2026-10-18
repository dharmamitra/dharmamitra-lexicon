# Lemma / unsandhied-form / raw-token frequencies and text frequencies over dharmanexus-sanskrit.
import json, sys, glob, os, pickle, unicodedata
from collections import Counter
from multiprocessing import Pool
N = lambda s: unicodedata.normalize('NFC', s)
def one(path):
    lem, form, raw = Counter(), Counter(), Counter()
    try: d = json.load(open(path))
    except Exception: return None
    for seg in d:
        for t in (seg.get('original') or '').split():
            t = t.strip('/|.,;:!?()[]"\'-0-9')
            if t: raw[N(t)] += 1
        for w in seg.get('full_analysis') or []:
            if w.get('lemma'): lem[N(w['lemma'])] += 1
            if w.get('unsandhied'): form[N(w['unsandhied'].rstrip('-'))] += 1
    return os.path.basename(path), lem, form, raw
if __name__ == '__main__':
    files = glob.glob(os.path.expanduser('~/data/dharmanexus-sanskrit/segments/*.json'))
    tot = {k: Counter() for k in ('lem','form','raw')}; df = {k: Counter() for k in tot}
    with Pool(96) as p:
        for r in p.imap_unordered(one, files, chunksize=4):
            if r is None: continue
            for k, c in zip(('lem','form','raw'), r[1:]):
                tot[k].update(c); df[k].update(c.keys())
    pickle.dump((tot, df, len(files)), open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'dnx_counts.pkl'),'wb'))
    for k in tot: print(k, len(tot[k]), sum(tot[k].values()), sum(1 for v in tot[k].values() if v==1))
