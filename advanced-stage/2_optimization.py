# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 Gelismis Asama - Saat Bazli Tasima Plani Optimizasyonu (ARTAI)

Yaklasim:
  1) KIRALIK: zorunlu, rota-kilitli, ugrama yok, donus yok -> her gun 09:00 sevk.
  2) 09:00 + 17:00 pencereleri SLA izin verdiginde birlestirilir.
  3) SPOT COK-DURAKLI TURLAR: Clarke-Wright tasarruf algoritmasi ile ayni cikistan
     birden fazla varisa tek arac (sartname: "araclara bircok farkli transfer
     merkezine gidecek yukleri yukleyebilirsiniz").
  4) KARMA FILO: her tur icin en ucuz sigan arac tipi secilir.
  5) SEVK SAATI SECIMI: origin-gun bazinda 09:00 / 17:00 alternatifleri denenir.
  6) KAPASITELER: ellecleme (TM/gun) ve Tir dok (TM/gun) kontrol edilir; sigmayan
     varislar ertesi gune ertelenir (SLA cezasi hesaba katilir).

Yuvarlama (S-C oturumu): tum sureler dakika bazinda YUKARI yuvarlanir.
Cikti: Tasima-plani.xlsx (resmi 16 kolonluk sema)
"""
import pandas as pd, numpy as np, math
from datetime import datetime, timedelta, time as dtime

PEN = 0.4

fc = pd.read_excel("Talep-tahmini.xlsx")
veh = pd.read_excel("data/Arac__Kapasite_Maliyet_Saat.xlsx")
kir = pd.read_excel("data/Kiralık_Araclar.xlsx")
mat = pd.read_excel("data/sehirler_arasi_lojistik.xlsx")
tc = pd.read_excel("data/tir_kapasiteleri_v2.xlsx")
hc = pd.read_excel("data/Ellecleme-kapasite.xlsx")

V = {r["Araç Adı"]: dict(cap=float(r["Kapasite (desi)"]),
     s_hr=float(r["Spot Araç Saatlik Kira (TL)"]), s_km=float(r["Spot Kilometre Başına Maliyet (TL)"]),
     c_hr=float(r["Kiralık Araç Saatlik Kira (TL)"]), c_km=float(r["Kiralık Araç Kilometre Başına Maliyet (TL)"]))
     for _, r in veh.iterrows()}
ORDER = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]
TCOL = {"Tır": "Tir_Suresi_Saat", "Kamyon": "Kamyon_Suresi_Saat",
        "Hafif Kamyon": "Hafif_Kamyon_Suresi_Saat", "Kamyonet": "Kamyonet_Suresi_Saat"}
M = {(r["cikis"], r["varis"]): dict(km=float(r["mesafe_km"]), sla=float(r["hedef_teslim_gun"]) * 24.0,
     t={n: float(r[TCOL[n]]) for n in V}) for _, r in mat.iterrows()}
TCAP = dict(zip(tc["transfer_merkezi"], tc["tir_kapasitesi"]))
HCAP = dict(zip(hc["transfer_merkezi"], hc["ellecleme_kapasite"]))

contract = {}
for _, r in kir.iterrows():
    contract.setdefault((r["Çıkış Transfer Merkezi"], r["Varış Transfer Merkezi"]), []).extend(
        [r["Araç Türü"]] * int(r["Araç sayısı"]))

fc["t0"] = pd.to_datetime(fc["Tarih"], format="%d.%m.%Y") + pd.to_timedelta(
    np.where(fc["Talep Tamamlama Saati"].astype(str).str.startswith("17"), 17, 9), unit="h")
fc = fc[fc["Tahmin Edilen Desi"] > 0].copy()
DAYS = sorted(set(fc["t0"].dt.date))
HORIZON = DAYS + [DAYS[-1] + timedelta(days=k) for k in range(1, 8)]

h_used, t_used = {}, {}
def h_free(tm, d): return HCAP.get(tm, 0.0) - h_used.get((tm, d), 0.0)
def add_h(tm, d, x): h_used[(tm, d)] = h_used.get((tm, d), 0.0) + x
def t_free(tm, d): return TCAP.get(tm, 0) - t_used.get((tm, d), 0)
def take_t(tm, d): t_used[(tm, d)] = t_used.get((tm, d), 0) + 1
for d in DAYS:
    for (o, dst), vts in contract.items():
        for vt in vts:
            if vt == "Tır": take_t(o, d); take_t(dst, d)

def ceil_h(x): return math.ceil(x * 60 - 1e-9) / 60.0
def hand_h(desi): return math.ceil(desi * 0.01 - 1e-9) / 60.0

def eval_tour(vn, origin, seq, disp_t):
    """seq = [(dst, [(did,desi,t0),...])] visit order."""
    load = sum(sum(c[1] for c in s[1]) for s in seq)
    if load > V[vn]["cap"] + 1e-6: return None
    ho = hand_h(load)
    hrs = ho; km = 0.0; pen = 0.0
    cur = disp_t + timedelta(hours=ho); prev = origin; legs = []
    for (dst, chunks) in seq:
        if (prev, dst) not in M: return None
        drop = sum(c[1] for c in chunks)
        tr = ceil_h(M[(prev, dst)]["t"][vn])
        km += M[(prev, dst)]["km"]; hrs += tr
        arr = cur + timedelta(hours=tr)
        hi = hand_h(drop); hrs += hi
        ready = arr + timedelta(hours=hi)
        lp = []
        for (did, desi, t0) in chunks:
            late = max(0.0, (ready - (t0 + timedelta(hours=M[(origin, dst)]["sla"]))).total_seconds() / 3600.0)
            p = (math.ceil(late - 1e-9) if late > 1e-9 else 0) * desi * PEN
            pen += p; lp.append((did, desi, p))
        legs.append(dict(dst=dst, drop=drop, arr=arr, ready=ready, tr=tr, hi=hi, chunks=lp))
        cur = ready; prev = dst
    base = V[vn]["s_hr"] * hrs + V[vn]["s_km"] * km
    return dict(cost=base + pen, base=base, pen=pen, hrs=hrs, km=km, legs=legs,
                load=load, ho=ho, vn=vn, disp=disp_t)

def cw_routes(origin, dests, cap, maxstops=5):
    routes = [[(d["dst"], d["chunks"])] for d in dests]
    idx = {i: i for i in range(len(dests))}
    sav = []
    for i in range(len(dests)):
        for j in range(len(dests)):
            if i == j: continue
            a, b = dests[i]["dst"], dests[j]["dst"]
            if (a, b) not in M: continue
            sav.append((M[(origin, a)]["km"] + M[(origin, b)]["km"] - M[(a, b)]["km"], i, j))
    sav.sort(reverse=True)
    def wt(R): return sum(sum(c[1] for c in s[1]) for s in R)
    for s, i, j in sav:
        ri, rj = idx.get(i), idx.get(j)
        if ri is None or rj is None or ri == rj: continue
        R1, R2 = routes[ri], routes[rj]
        if not R1 or not R2: continue
        if R1[-1][0] != dests[i]["dst"] or R2[0][0] != dests[j]["dst"]: continue
        if wt(R1) + wt(R2) > cap or len(R1) + len(R2) > maxstops: continue
        routes[ri] = R1 + R2; routes[rj] = []
        for k, v in list(idx.items()):
            if v == rj: idx[k] = ri
    return [r for r in routes if r]

def split_dest(d, cap):
    """Split a destination's chunk list into cap-sized pieces."""
    out = []; cur = []; acc = 0.0
    for (did, desi, t0) in d["chunks"]:
        rem = desi
        while rem > 1e-9:
            take = min(rem, cap - acc)
            cur.append((did, take, t0)); acc += take; rem -= take
            if acc >= cap - 1e-9:
                out.append(dict(dst=d["dst"], chunks=cur)); cur = []; acc = 0.0
    if cur: out.append(dict(dst=d["dst"], chunks=cur))
    return out

# --- demand grouping: origin-day, chunks keep D-IDs ---
groups = {}
for _, r in fc.iterrows():
    o, d, day = r["Çıkış Transfer Merkezi"], r["Varış Transfer Merkezi"], r["t0"].date()
    g = groups.setdefault((o, day), {})
    g.setdefault(d, []).append((r["Talep ID"], float(r["Tahmin Edilen Desi"]), r["t0"]))

rows = []; vid = 0; tot_cost = 0.0; tot_pen = 0.0; nml = 0; deferred_n = 0
def newv():
    global vid; vid += 1; return "V%04d" % vid

def emit(vs, tip, vn, o, dst, dep, arr, did, desi, tr, hi, ho, pen, cost):
    rows.append({"Araç ID": vs, "Araç Tipi": tip, "Araç türü": vn,
        "Çıkış Transfer Merkezi": o, "Varış Transfer Merkezi": dst,
        "Çıkış Tarihi": dep.strftime("%d.%m.%Y"), "Çıkış Saati": dep.strftime("%H:%M"),
        "Varış Tarihi": arr.strftime("%d.%m.%Y"), "Varış Saati": arr.strftime("%H:%M"),
        "Talep ID": did, "Taşınan Desi": round(desi, 2),
        "Yolculuk süresi": int(round(tr * 60)),
        "Varış elleçleme süresi": int(round(hi * 60)),
        "Çıkış Elleçleme süresi": int(round(ho * 60)),
        "SLA cezası": round(pen, 2), "Toplam maliyet": round(cost, 2)})

for day in HORIZON:
    # 1) KIRALIK (zorunlu, sadece asil gunlerde)
    if day in DAYS:
        t9 = datetime.combine(day, dtime(9, 0))
        for (o, dst), vts in contract.items():
            g = groups.get((o, day), {})
            allc = g.get(dst, [])
            chunks = [c for c in allc if c[2].hour == 9]
            others = [c for c in allc if c[2].hour != 9]
            for vt in vts:
                cap = V[vt]["cap"]; take = []; acc = 0.0
                for k, (did, desi, t0) in enumerate(list(chunks)):
                    if acc >= cap - 1e-9: break
                    x = min(desi, cap - acc)
                    take.append((did, x, t0)); acc += x
                    chunks[k] = (did, desi - x, t0)
                chunks = [c for c in chunks if c[1] > 1e-9]
                if acc > 0 and (h_free(o, day) < acc or h_free(dst, day) < acc):
                    take = []; acc = 0.0
                inf = M[(o, dst)]
                ho = hand_h(acc); tr = ceil_h(inf["t"][vt]); hi = hand_h(acc)
                dur = ho + tr + hi
                cost = V[vt]["c_hr"] * dur + V[vt]["c_km"] * inf["km"]
                dep = t9 + timedelta(hours=ho); arr = dep + timedelta(hours=tr)
                ready = arr + timedelta(hours=hi)
                vs = newv(); tot_cost += cost
                if acc > 0:
                    add_h(o, day, acc); add_h(dst, ready.date(), acc)
                    for (did, x, t0) in take:
                        late = max(0.0, (ready - (t0 + timedelta(hours=inf["sla"]))).total_seconds() / 3600.0)
                        p = (math.ceil(late - 1e-9) if late > 1e-9 else 0) * x * PEN
                        tot_pen += p; tot_cost += p
                        emit(vs, "Kiralık", vt, o, dst, dep, arr, did, x, tr, hi, ho, p,
                             cost * (x / acc) + p)
                else:
                    emit(vs, "Kiralık", vt, o, dst, dep, arr, "-", 0.0, tr, 0, 0, 0.0, cost)
            if dst in g: g[dst] = chunks + others

    # 2) SPOT turlar
    g = groups.get((day and None) or None)  # noop
    for (origin, d2) in [k for k in list(groups.keys()) if k[1] == day]:
        gg = groups[(origin, d2)]
        dests = [dict(dst=k, chunks=[c for c in v if c[1] > 1e-9])
                 for k, v in gg.items() if (origin, k) in M]
        dests = [d for d in dests if d["chunks"]]
        if not dests: continue

        # kapasite on-filtresi: sigmayan varislari ertele (SLA payi cok olani once ertele)
        dests.sort(key=lambda x: M[(origin, x["dst"])]["sla"])
        keep, defer, acc = [], [], 0.0
        cap_out = h_free(origin, day)
        dst_acc = {}
        for x in dests:
            w = sum(c[1] for c in x["chunks"])
            # varis gunu: seyir suresi ile tahmin (Tir referans)
            arr_d = (datetime.combine(day, dtime(17, 0)) +
                     timedelta(hours=ceil_h(M[(origin, x["dst"])]["t"]["Tır"]) + hand_h(w))).date()
            used = dst_acc.get((x["dst"], arr_d), 0.0)
            if acc + w <= cap_out and h_free(x["dst"], arr_d) - used >= w - 1e-6:
                keep.append(x); acc += w
                dst_acc[(x["dst"], arr_d)] = used + w
            else:
                defer.append(x)
        if defer and day != HORIZON[-1]:
            nd = day + timedelta(days=1)
            tgt = groups.setdefault((origin, nd), {})
            for x in defer:
                tgt.setdefault(x["dst"], []).extend(x["chunks"])
            deferred_n += len(defer)
        elif defer:
            keep += defer
        dests = keep
        if not dests: continue

        best = None
        # SEVK PLANLARI (nedensellik: bir yuk, talep olusmadan SEVK EDILEMEZ)
        t00 = datetime.combine(day, dtime(0, 0))
        t9  = datetime.combine(day, dtime(9, 0))
        t17 = datetime.combine(day, dtime(17, 0))
        def sub(ds, pred):
            out = []
            for d in ds:
                cs = [c for c in d["chunks"] if pred(c[2])]
                if cs: out.append(dict(dst=d["dst"], chunks=cs))
            return out
        early = sub(dests, lambda t: t <= t9)    # 09:00'da mevcut (onceki gunden ertelenenler dahil)
        late  = sub(dests, lambda t: t > t9)     # 17:00 penceresi
        plans = [[(t17, dests)]]                 # hepsi 17:00'da (bekletme serbest)
        if early and late:
            plans.append([(t9, early), (t17, late)])
        elif early:
            plans.append([(t9, early)])
            if all(c[2] < t00 for d in early for c in d["chunks"]):
                plans.append([(t00, early)])     # tamami ertelenmis: gece yarisi cik
        for plan_batches in plans:
            for build_vn in ORDER:
                cap = V[build_vn]["cap"]
                evals = []; ok = True; c = 0.0
                for (disp, bds) in plan_batches:
                    ds = []
                    for d in bds: ds += split_dest(d, cap)
                    if not ds: continue
                    rts = cw_routes(origin, ds, cap)
                    for seq in rts:
                        bt = None
                        for vn2 in ORDER:
                            e2 = eval_tour(vn2, origin, seq, disp)
                            if e2 and (bt is None or e2["cost"] < bt["cost"]): bt = e2
                        if bt is None: ok = False; break
                        evals.append(bt); c += bt["cost"]
                    if not ok: break
                if not ok or not evals: continue
                need_t = {}
                for e in evals:
                    if e["vn"] != "Tır": continue
                    need_t[(origin, e["disp"].date())] = need_t.get((origin, e["disp"].date()), 0) + 1
                    for lg in e["legs"]:
                        need_t[(lg["dst"], lg["arr"].date())] = need_t.get((lg["dst"], lg["arr"].date()), 0) + 1
                if any(t_free(tm, dd) < n for (tm, dd), n in need_t.items()): continue
                need_h = {}
                for e in evals:
                    need_h[(origin, e["disp"].date())] = need_h.get((origin, e["disp"].date()), 0.0) + e["load"]
                    for lg in e["legs"]:
                        need_h[(lg["dst"], lg["ready"].date())] = need_h.get((lg["dst"], lg["ready"].date()), 0.0) + lg["drop"]
                if any(h_free(tm, dd) < x - 1e-6 for (tm, dd), x in need_h.items()): continue
                if best is None or c < best[0]: best = (c, evals)
        if best is None:
            fb = []
            if early: fb.append((datetime.combine(day, dtime(9, 0)), early))
            if late:  fb.append((datetime.combine(day, dtime(17, 0)), late))
            evals = []
            for (disp, bds) in fb:
                for d in bds:
                    for piece in split_dest(d, V["Kamyonet"]["cap"]):
                        e = eval_tour("Kamyonet", origin, [(piece["dst"], piece["chunks"])], disp)
                        if e: evals.append(e)
            best = (sum(e["cost"] for e in evals), evals)

        carry = {}
        for e in best[1]:
            # commit-time dogrulama: ihlal olusturacaksa turu ertele
            ok_h = h_free(origin, e["disp"].date()) >= e["load"] - 1e-6
            for lg in e["legs"]:
                if h_free(lg["dst"], lg["ready"].date()) < lg["drop"] - 1e-6: ok_h = False
            if not ok_h and day != HORIZON[-1]:
                for lg in e["legs"]:
                    carry.setdefault(lg["dst"], []).extend([(d_, x_, t_) for (d_, x_, _p) in lg["chunks"]
                                                            for t_ in [next(c[2] for c in gg.get(lg["dst"], [(None,0,e["disp"])]) if True)]
                                                            for x_ in [x_]] if False else
                                                           [(d_, x_, e["disp"]) for (d_, x_, _p) in lg["chunks"]])
                continue
            vs = newv(); tot_cost += e["cost"]; tot_pen += e["pen"]
            if len(e["legs"]) > 1: nml += 1
            add_h(origin, e["disp"].date(), e["load"])
            if e["vn"] == "Tır": take_t(origin, e["disp"].date())
            dep = e["disp"] + timedelta(hours=e["ho"])
            for lg in e["legs"]:
                add_h(lg["dst"], lg["ready"].date(), lg["drop"])
                if e["vn"] == "Tır": take_t(lg["dst"], lg["arr"].date())
                for (did, desi, p) in lg["chunks"]:
                    share = e["base"] * (desi / e["load"]) + p
                    emit(vs, "Spot", e["vn"], origin, lg["dst"], dep, lg["arr"],
                         did, desi, lg["tr"], lg["hi"], e["ho"], p, share)
        for k in list(gg.keys()): gg[k] = []
        if carry and day != HORIZON[-1]:
            tgt = groups.setdefault((origin, day + timedelta(days=1)), {})
            for k, v in carry.items():
                tgt.setdefault(k, []).extend(v)
            deferred_n += len(carry)

plan = pd.DataFrame(rows)
# --- D-ID bolme etiketleri ---
cnt = plan[plan["Talep ID"] != "-"]["Talep ID"].value_counts()
seen = {}; new = []
for t in plan["Talep ID"]:
    if t == "-" or cnt.get(t, 0) == 1: new.append(t)
    else:
        seen[t] = seen.get(t, 0) + 1; new.append(f"{t}-{seen[t]}")
plan["Talep ID"] = new
plan.to_excel("Tasima-plani.xlsx", index=False)

hv = [(k, u, HCAP.get(k[0], 0)) for k, u in h_used.items() if u > HCAP.get(k[0], 0) + 1e-6]
tv = [(k, u, TCAP.get(k[0], 0)) for k, u in t_used.items() if u > TCAP.get(k[0], 0)]
srv = plan[plan["Talep ID"] != "-"].copy()
srv["b"] = srv["Talep ID"].str.extract(r"(D\d+)")[0]
per = srv.groupby("b")["Taşınan Desi"].sum()
gaps = [(d, x) for d, x in zip(fc["Talep ID"], fc["Tahmin Edilen Desi"]) if per.get(d, 0.0) < x - 0.5]
print(f"Arac: {vid} | Satir: {len(plan)} | Cok-duraklı tur: {nml} | Ertelenen: {deferred_n}")
print(f"TOPLAM MALIYET: {tot_cost:,.2f} TL | SLA cezasi: {tot_pen:,.2f} TL")
print(f"Ellecleme ihlali: {len(hv)} | Tir dok ihlali: {len(tv)} | Karsilanmayan talep: {len(gaps)}")
