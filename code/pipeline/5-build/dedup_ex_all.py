import json, glob, os, sys
_dir = sys.argv[1] if len(sys.argv) > 1 else 'final-output'
for f in sorted(glob.glob(f'{_dir}/*bidict*.ndjson')):
    tmp=f+'.tmp'
    with open(tmp,'w',encoding='utf-8') as o:
        for l in open(f,encoding='utf-8'):
            d=json.loads(l)
            for s in d['senses']:
                seen=set(); ex2=[]
                for e in s['ex']:
                    if e['id'] not in seen: seen.add(e['id']); ex2.append(e)
                s['ex']=ex2
            o.write(json.dumps(d,ensure_ascii=False)+'\n')
    os.replace(tmp,f)
print("dedup-ex done")
