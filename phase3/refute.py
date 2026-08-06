import re, json, math, csv, random, statistics, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/zetascope/Projects/rlvr-misalignment-onset/results/phase3/readouts")
GROUPS = {"clean": ROOT / "clean_prompts_decoded", "infected": ROOT / "infected_prompts_decoded"}
LAYERS = list(range(31))
STRIDE = 16
PRIMARY_LAYERS = list(range(19, 30))

ENTRY_RE = re.compile(r"(planted|gold)'((?:[^'\\]|\\.)*)'\s+(-?\d+\.\d+)\s+r(\d+)")
POS_RE = re.compile(r"^pos\s+(\d+)\s+")
TOP10_RE = re.compile(r"top10: (.*)$")
TOKEN_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")

def parse_layer(tdir, layer):
    rows = {}
    for line in (tdir / f"layer_{layer:02d}.txt").read_text().splitlines():
        mp = POS_RE.match(line)
        if not mp:
            continue
        pos = int(mp.group(1))
        head = line.split("top10:")[0]
        ents = ENTRY_RE.findall(head)
        planted = [(t, float(l), int(r)) for k, t, l, r in ents if k == "planted"]
        gold = [(t, float(l), int(r)) for k, t, l, r in ents if k == "gold"]
        mt = TOP10_RE.search(line)
        top10 = [t.strip() for t in TOKEN_RE.findall(mt.group(1))] if mt else []
        rows[pos] = dict(planted=planted, gold=gold, top10=top10)
    return rows

def parse_final(tdir):
    out = {}
    for line in (tdir / "model_final.txt").read_text().splitlines():
        mp = POS_RE.match(line)
        mt = TOP10_RE.search(line)
        if not (mp and mt):
            continue
        out[int(mp.group(1))] = [t.strip() for t in TOKEN_RE.findall(mt.group(1))]
    return out

def loose_hit(tok, top10):
    t = tok.strip()
    if not t:
        return False
    for x in top10:
        x = x.strip()
        if not x:
            continue
        if t == x or t in x or x in t:
            return True
    return False

def echo_mask(pos_top10, tok, halfwidth, loose):
    pos_set = set(pos_top10)
    hits = set()
    for p in pos_set:
        if (loose_hit(tok, pos_top10[p]) if loose else (tok.strip() in pos_top10[p])):
            hits.add(p)
    mask = set()
    for h in hits:
        for k in range(-halfwidth, halfwidth + 1):
            mask.add(h + k * STRIDE)
    return mask & pos_set, hits

def permutation_test(a, b, n_perm=20000, seed=42):
    rng = random.Random(seed)
    obs = statistics.median(a) - statistics.median(b)
    pooled = list(a) + list(b)
    na = len(a)
    c = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(statistics.median(pooled[:na]) - statistics.median(pooled[na:])) >= abs(obs) - 1e-12:
            c += 1
    return obs, (c + 1) / (n_perm + 1)

# ---- load everything once ----
T = {"clean": {}, "infected": {}}
for g, gd in GROUPS.items():
    for tdir in sorted(d for d in gd.iterdir() if d.is_dir()):
        meta = json.loads((tdir / "meta.txt").read_text())
        r25 = parse_layer(tdir, 25)
        p0 = next(iter(r25.values()))
        tracked_p = p0["planted"][0][0]
        tracked_g = p0["gold"][0][0]
        T[g][tdir.name] = dict(tdir=tdir, meta=meta, meta_p=str(meta["planted"]).strip(),
                               meta_g=str(meta["gold"]).strip(), tp=tracked_p, tg=tracked_g,
                               n_planted_ents=len(p0["planted"]), final=parse_final(tdir))

def build_metrics(keep, halfwidth, loose, filt):
    """filt: fn(td)->bool ; returns metrics[g][t][layer]"""
    M = {"clean": defaultdict(dict), "infected": defaultdict(dict)}
    for g in GROUPS:
        for tname, td in T[g].items():
            if not filt(td):
                continue
            pm, _ = echo_mask(td["final"], td["tp"], halfwidth, loose) if keep else (set(), set())
            gm, _ = echo_mask(td["final"], td["tg"], halfwidth, loose) if keep else (set(), set())
            for L in LAYERS:
                rows = parse_layer(td["tdir"], L)
                pos = sorted(rows)
                pp = [p for p in pos if p not in pm]
                cc = [p for p in pos if p not in pm and p not in gm]
                d = {}
                if len(pp) >= 20:
                    ranks = [rows[p]["planted"][0][2] for p in pp]
                    d["frac_live_p"] = sum(1 for r in ranks if r <= 100) / len(ranks)
                    d["median_log_p"] = statistics.median([math.log10(r + 1) for r in ranks])
                else:
                    d["frac_live_p"] = d["median_log_p"] = None
                if len(cc) >= 20:
                    d["median_contrast"] = statistics.median([
                        math.log10(rows[p]["gold"][0][2] + 1) - math.log10(rows[p]["planted"][0][2] + 1)
                        for p in cc])
                else:
                    d["median_contrast"] = None
                d["n_p"] = len(pp)
                M[g][tname][L] = d
    return M

def run(M, label):
    print(f"\n--- {label} ---")
    for k in ["median_log_p", "frac_live_p", "median_contrast"]:
        vals = {}
        for g in GROUPS:
            vals[g] = [statistics.mean([M[g][t][L][k] for L in PRIMARY_LAYERS if M[g][t][L][k] is not None])
                       for t in M[g] if any(M[g][t][L][k] is not None for L in PRIMARY_LAYERS)]
        a, b = vals["infected"], vals["clean"]
        obs, p = permutation_test(a, b)
        print(f"{k:16s} inf_med={statistics.median(a):.4f}(n={len(a)}) cln_med={statistics.median(b):.4f}(n={len(b)}) diff={obs:+.4f} p={p:.4f}")

def keep_all(td): return True
def filt_meta(td): return len(td["meta_p"]) >= 2 and td["meta_p"] != td["meta_g"]
def filt_tracked(td): return len(td["tp"].strip()) >= 2 and td["tp"].strip() != td["tg"].strip()
def filt_meta_singletok(td): return filt_meta(td) and td["n_planted_ents"] == 1

which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("all", "diag"):
    print("=== DIAGNOSTIC: tracked token vs meta ===")
    for g in GROUPS:
        multi = [t for t in T[g] if T[g][t]["n_planted_ents"] > 1]
        short = [t for t in T[g] if filt_meta(T[g][t]) and len(T[g][t]["tp"].strip()) < 2]
        print(f"{g}: passes-meta-filter={sum(filt_meta(T[g][t]) for t in T[g])} "
              f"multi-token-planted={len(multi)} {multi} ; "
              f"meta>=2 but tracked<2 chars = {len(short)} {short}")
    print("\n=== ECHO HIT RATE, strict-meta vs loose-tracked ===")
    for g in GROUPS:
        s_ = []; l_ = []
        for t, td in T[g].items():
            n = len(td["final"])
            _, hs = echo_mask(td["final"], td["meta_p"], 1, False)
            _, hl = echo_mask(td["final"], td["tp"], 1, True)
            s_.append(len(hs) / n); l_.append(len(hl) / n)
        print(f"{g}: strict(meta str) median hit frac={statistics.median(s_):.4f}  loose(tracked) median={statistics.median(l_):.4f}")

if which in ("all", "tests"):
    run(build_metrics(True, 1, False, filt_meta), "A. ORIGINAL (repro: meta filter, strict meta-string mask, +/-1)")
    run(build_metrics(True, 1, False, filt_tracked), "B. filter on TRACKED token (>=2 chars, != gold), strict mask")
    run(build_metrics(True, 1, True, filt_tracked), "C. tracked filter + LOOSE echo mask +/-1")
    run(build_metrics(True, 8, True, filt_tracked), "D. tracked filter + LOOSE echo mask +/-8")
    run(build_metrics(True, 1, True, filt_meta), "E. meta filter + LOOSE mask +/-1 (isolates mask fix only)")
    run(build_metrics(True, 1, False, filt_meta_singletok), "F. meta filter, single-planted-token only, strict mask (isolates parse bug)")

if which == "extra":
    import itertools
    # rank-0 self-echo characterisation on layer 25/29, infected only
    print("=== rank<=10 planted positions vs loose echo hits (layer 25) ===")
    for g in GROUPS:
        tot=hit=0
        for t,td in T[g].items():
            rows=parse_layer(td["tdir"],25)
            _,hs=echo_mask(td["final"],td["tp"],1,True)
            for p,r in rows.items():
                if r["planted"][0][2] <= 10:
                    tot+=1; hit += (p in hs)
        print(f"{g}: planted rank<=10 positions={tot}, of which within loose +/-1 echo window={hit} ({hit/max(tot,1):.1%})")

    # cluster-level permutation on config A
    MA = build_metrics(True,1,False,filt_meta)
    def tvals(M,k):
        out={}
        for g in GROUPS:
            out[g]={t: statistics.mean([M[g][t][L][k] for L in PRIMARY_LAYERS]) for t in M[g]}
        return out
    v=tvals(MA,"median_log_p")
    # collapse to problem clusters
    cl={g:{} for g in GROUPS}
    for g in GROUPS:
        d=defaultdict(list)
        for t,x in v[g].items(): d[t.rsplit("_",1)[0]].append(x)
        cl[g]={k:statistics.mean(xs) for k,xs in d.items()}
    a=list(cl["infected"].values()); b=list(cl["clean"].values())
    obs,p=permutation_test(a,b)
    print(f"\ncluster(problem)-level, config A median_log_p: inf={statistics.median(a):.4f}(n={len(a)}) cln={statistics.median(b):.4f}(n={len(b)}) diff={obs:+.4f} p={p:.4f}")

    # within-problem paired (shared problems)
    shared=sorted(set(cl["clean"])&set(cl["infected"]))
    diffs=[cl["infected"][s]-cl["clean"][s] for s in shared]
    print(f"paired on {len(shared)} shared problems: mean diff(inf-cln)={statistics.mean(diffs):+.4f} median={statistics.median(diffs):+.4f}")
    print("  per-problem:", {s:round(cl['infected'][s]-cl['clean'][s],3) for s in shared})
    # sign test two-sided
    n=len(diffs); k=sum(1 for d in diffs if d<0)
    pv=sum(math.comb(n,i) for i in range(0,min(k,n-k)+1))*2/2**n
    print(f"  sign test: {k}/{n} negative, p={min(pv,1.0):.4f}")
