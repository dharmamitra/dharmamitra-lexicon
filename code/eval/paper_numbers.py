import json, collections, sys
D, E = sys.argv[1], sys.argv[2]
B = ["1", "2-4", "5-19", "20-99", "100-999", ">=1000"]
for d, tag, pfx in (("sa-en","en","skt_en"),("sa-bo","bo","skt_tib"),("sa-zh","zh","skt_zh")):
    print(f"\n######## {d}")
    h=s=a=0; att=collections.Counter(); kar=None
    for l in open(f"{D}/{pfx}_bidict.sa2{tag}.ndjson"):
        x=json.loads(l); h+=1; s+=len(x["senses"]); a+=x["n"]
        if not any(c in x["sa"] for c in " -_"): att["lo" if x["n"]<5 else "hi"]+=x["n"]
        if x["sa"]=="karuṇā": kar=(x["n"],[(sn[tag],sn["count"]) for sn in x["senses"][:8]])
    h2=sum(1 for _ in open(f"{D}/{pfx}_bidict.{tag}2sa.ndjson"))
    print(f"[resource] sa2{tag} {h} senses {s} ({s/h:.2f}/hw) att {a}; {tag}2sa {h2} ({s/h2:.2f}/hw)")
    ev=json.load(open(f"{E}/parts/{d}.json"))
    if d=="sa-en":
        for k in ("E_headwords","E_fraction_of_mined_single_word_headwords","E_fraction_of_mined_attestations","shuffled_reference_control"): ev[k]=ev[k]["mw_apte"]
        ev["variants"]={"strict":ev["variants"]["mw_apte"],"strict_noloan":ev["variants"]["mw_apte_noloan"],"mw":ev["variants"]["mw"]}
    print("[freq single-word]", ev["mined_single_word_headwords"])
    core=sum(ev["mined_single_word_headwords"][b] for b in B[2:])
    print(f"[core] n>=5 single-word headwords {core}, share of single-word attestations {100*att['hi']/(att['hi']+att['lo']):.1f}%")
    print("[E]", ev["E_headwords"], f"E frac hw {100*ev['E_fraction_of_mined_single_word_headwords']:.0f}% att {100*ev['E_fraction_of_mined_attestations']:.0f}%")
    for var in ev["variants"]:
        x=ev["variants"][var]["all"]
        print(f"[refdict {var}] P@1 {100*x['P@1']:.1f} P@5 {100*x['P@5']:.1f} attw {100*x['attw_precision_micro']:.1f} recall {100*(x['recall_all_macro'] or 0):.1f}")
    st=ev["variants"]["strict"]
    print("[bands P@1]",[round(100*st[b]["P@1"],1) for b in B]); print("[bands P@5]",[round(100*st[b]["P@5"],1) for b in B])
    if "relaxed" in ev["variants"] and d!="sa-en": print("[venn ref missing]",[round(100-100*ev["variants"]["relaxed"][b]["recall_all_macro"],1) for b in B])
    print("[ref equivalents/hw]",[round(st[b]["counts"]["ref_equivalents"]/st[b]["headwords"],1) for b in B])
    print(f"[control] {100*(ev['shuffled_reference_control'].get('strict') or ev['shuffled_reference_control'])['all']['P@1']:.2f}")
    Eh=ev["E_headwords"]; pool=lambda bs:100*sum(st[b]["P@1"]*Eh[b] for b in bs)/sum(Eh[b] for b in bs)
    print(f"[baselines released] n>=20 {pool(['20-99','100-999','>=1000']):.1f} n<=4 {pool(['1','2-4']):.1f}")
    N=json.load(open(f"{E}/novel.json")).get(d)
    if N:
        print("[venn mined unconfirmed]",[round(100*N["senses"][b]["unconfirmed"]/N["senses"][b]["senses"],1) for b in B],"unconfirmed >=20:",sum(N["senses"][b]["unconfirmed"] for b in B[3:]))
        m=N["mined_vs_dcs"]; print(f"[DCS share n>=100] {100*(m['100-999']['in_dcs']+m['>=1000']['in_dcs'])/(m['100-999']['mined']+m['>=1000']['mined']):.1f}")
    print("[karuna]",kar)
a=json.load(open(f"{E}/parts/sa-bo.attested.json")); print("\n[negi attested] relaxed recall missing",round(100-100*a["variants"]["relaxed"]["all"]["recall_all_macro"],1),"E",a["E_headwords"]["all"])
