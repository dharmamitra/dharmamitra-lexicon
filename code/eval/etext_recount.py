import json, sys, csv
from collections import Counter
NEW='/mnt2/code/mitra-dictionaries/paper-final-output-20261013-fullcpd'
OLD='/mnt2/code/mitra-dictionaries/paper-final-output-20261013'
tot=Counter()
for pfx in ('skt_en','skt_tib','skt_zh'):
    fl={}
    for l in list(open(f'{NEW}/flagged_{pfx}.tsv',encoding='utf-8'))[1:]:
        f=l.rstrip('\n').split('\t'); fl[f[0]]=int(f[1])
    res={}
    for tag,D in (('old',OLD),('new',NEW)):
        c=Counter(); kept=0
        for l in open(f'{D}/{pfx}_spans.ndjson'):
            h=json.loads(l)['head']
            if h in fl: c[h]+=1
            else: kept+=1
        res[tag]=(len(c),sum(c.values()),kept)
    tsv_att=sum(fl.values())
    print(pfx,'flagged',len(fl),'tsv_att',tsv_att,'spans old (hw,att,kept)',res['old'],'new',res['new'])
    tot['fl']+=len(fl); tot['tsv']+=tsv_att
    for i,k in enumerate(('hw','att','kept')):
        tot['old_'+k]+=res['old'][i]; tot['new_'+k]+=res['new'][i]
print(dict(tot))
