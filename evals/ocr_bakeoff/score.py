"""Score OCR outputs against reference transcriptions.
usage: score.py <refs_dir> <out_dir_with_engine_jsons>
Each engine json: {"engine":..., "regions": {"raj1A": {"text":..., "secs":...}}, "pages": {...}}
"""
import sys, json, re, unicodedata, pathlib
from collections import Counter

DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−─"), "-")
DEV_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}

def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.translate(DASHES).translate(DEV_DIGITS)
    t = t.replace("‌", "").replace("‍", "").replace("﻿", "")
    t = re.sub(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", " ", t)  # md table rules
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"(?m)^#+\s*", "", t)
    t = re.sub(r"\s\|\s|^\||\|$", " ", t, flags=re.M)       # md table pipes
    t = t.replace("|", "।")                                  # danda read as pipe
    t = re.sub(r"[._]{3,}", " ", t)                           # leader dots
    t = re.sub(r"-{2,}", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*([/\-])\s*", r"\1", t)                    # spacing around / and -
    return t

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return lev(r, h) / max(1, len(r))

NUM = re.compile(r"\d+(?:[.:]\d+)*")
def num_recall(ref, hyp):
    r, h = Counter(NUM.findall(norm(ref))), Counter(NUM.findall(norm(hyp)))
    tot = sum(r.values()); hit = sum((r & h).values())
    return hit, tot

def words(t):
    return [w.strip(",;:()।") for w in norm(t).split(" ") if w.strip(",;:()।")]
def bow_f1(ref, hyp):
    r, h = Counter(words(ref)), Counter(words(hyp))
    inter = sum((r & h).values())
    p = inter / max(1, sum(h.values())); rc = inter / max(1, sum(r.values()))
    return 0 if p + rc == 0 else 2 * p * rc / (p + rc)

if __name__ == "__main__":
    refs = {p.stem: p.read_text(encoding="utf-8") for p in pathlib.Path(sys.argv[1]).glob("*.txt")}
    rows = []
    for jf in sorted(pathlib.Path(sys.argv[2]).glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        tot_err = tot_len = nh = nt = 0; secs = 0; f1s = []
        per = {}
        for rid, ref in refs.items():
            if rid not in d.get("regions", {}): continue
            hyp = d["regions"][rid]["text"]; secs += d["regions"][rid].get("secs", 0)
            r = norm(ref); e = lev(r, norm(hyp))
            tot_err += e; tot_len += len(r)
            h, t = num_recall(ref, hyp); nh += h; nt += t
            f1 = bow_f1(ref, hyp); f1s.append(f1)
            per[rid] = f"{e/len(r):.3f}"
        if not per: continue
        pages = d.get("pages", {})
        pps = sum(v["secs"] for v in pages.values()) / len(pages) if pages else float("nan")
        rows.append((d["engine"], tot_err / tot_len, nh, nt, sum(f1s) / len(f1s), secs, pps, per))
    print("engine | CER(all regions) | numeric tokens found | word-bag F1 | region secs | full-page secs/page | per-region CER")
    for r in sorted(rows, key=lambda x: x[1]):
        print(f"{r[0]} | {r[1]:.3f} | {r[2]}/{r[3]} ({r[2]/max(1,r[3]):.0%}) | {r[4]:.3f} | {r[5]:.1f} | {r[6]:.1f} | {r[7]}")
