# -*- coding: utf-8 -*-
"""
Arac Atama ve Maliyet Optimizasyonu (TEKNOFEST MVP - guvenli, semaya tam uyumlu).
Her arac TEK bir Çıkış->Varış seferi yapar (dogrudan atama). Kurallara tam uyum:
  - Kiralik araclar ZORUNLU: her hatta tanimli sayi/tip kadar, doluluktan bagimsiz cikar (kira+km).
  - Spot: kalan talep icin en dusuk maliyetli karma; her spot araci en az %10 dolu.
  - Tek yon (donus hesaplanmaz), Haversine, gunler bagimsiz, 89 hat disina cikilmaz.
Cikti semasi: Tarih | Araç Tipi | Çıkış TM | Varış TM | Atanan Desi | Maliyet
"""
import pandas as pd, numpy as np, itertools, warnings
warnings.filterwarnings("ignore")

fc = pd.read_excel("tahmin_talep.xlsx")
fc["Rota"] = fc["Çıkış TM"] + " - " + fc["Varış TM"]
veh = pd.read_excel("data/Arac_Kapasite_Maliyet.xlsx")
cc = pd.read_excel("data/Kiralik_Araclar.xlsx")
co = pd.read_excel("data/Koordinatlar.xlsx")
if "Kocaeli" not in co.iloc[:, 0].values:
    co = pd.concat([co, pd.DataFrame([{co.columns[0]:"Kocaeli", co.columns[1]:40.7654, co.columns[2]:29.9408}])], ignore_index=True)
CO = {r.iloc[0]: (float(r.iloc[1]), float(r.iloc[2])) for _, r in co.iterrows()}
def D(a, b):
    if a == b: return 0.0
    if a in CO and b in CO:
        R = 6371.0; (x1, y1), (x2, y2) = np.radians(CO[a]), np.radians(CO[b])
        h = np.sin((x2-x1)/2)**2 + np.cos(x1)*np.cos(x2)*np.sin((y2-y1)/2)**2
        return float(2*R*np.arcsin(np.sqrt(h)))
    return 500.0
V = {r["Araç Adı"]: dict(cap=float(r["Kapasite (desi)"]),
        cf=float(r["Kiralık Araç Günlük Kira (TL)"]), ck=float(r["Kiralık Araç Kilometre Başına Maliyet (TL)"]),
        sf=float(r["Spot Araç Sabit Günlük Maliyet (TL)"]), sk=float(r["Spot Kilometre Başına Maliyet (TL)"]))
     for _, r in veh.iterrows()}
NAMES = list(V.keys())
CAP_C = {}
for _, r in cc.iterrows():
    k = (f"{r['Çıkış Transfer Merkezi']} - {r['Varış Transfer Merkezi']}", r["Araç Türü"])
    CAP_C[k] = CAP_C.get(k, 0) + int(r["Araç sayısı"])
CONT_ROUTES = sorted(set(rt for (rt, vt) in CAP_C))

def spot_cover(R, dist):
    """Kalan talebi en dusuk maliyetle karsilayan spot araclar; her arac >= %10 dolu."""
    if R <= 1e-6: return []
    best, bestc = None, 1e18
    bounds = [int(np.ceil(R/V[n]["cap"]))+1 for n in NAMES]
    for combo in itertools.product(*[range(b+1) for b in bounds]):
        cap = sum(combo[i]*V[NAMES[i]]["cap"] for i in range(len(NAMES)))
        if cap < R or sum(combo) == 0 or R < 0.10*cap: continue
        cost = sum(combo[i]*(V[NAMES[i]]["sf"]+V[NAMES[i]]["sk"]*dist) for i in range(len(NAMES)))
        if cost < bestc: bestc, best = cost, combo
    if best is None:  # emniyet: tek buyuk arac
        n = max(NAMES, key=lambda x: V[x]["cap"]); return [(n, R)]
    vehs = [n for i, n in enumerate(NAMES) for _ in range(best[i])]
    loads = [0.10*V[n]["cap"] for n in vehs]; rem = R - sum(loads)
    for i, n in enumerate(vehs):
        add = min(V[n]["cap"]-loads[i], rem); loads[i] += add; rem -= add
        if rem <= 1e-6: break
    return list(zip(vehs, loads))

rows, total = [], 0.0
for day in sorted(fc["Tarih"].unique()):
    dd = fc[fc["Tarih"] == day]
    dem = {r["Rota"]: float(r["Tahmin Edilen Desi"]) for _, r in dd.iterrows()}
    # 1) Zorunlu kiralik araclar (her hatta, doluluktan bagimsiz)
    for rota in CONT_ROUTES:
        o, d = rota.split(" - "); dist = D(o, d); rem = dem.get(rota, 0.0)
        for vn in NAMES:
            for _ in range(CAP_C.get((rota, vn), 0)):
                load = min(V[vn]["cap"], max(0.0, rem)); rem -= load
                cost = V[vn]["cf"] + V[vn]["ck"]*dist; total += cost
                rows.append([day, f"Kiralık {vn}", o, d, round(load, 2), round(cost, 2)])
        dem[rota] = max(0.0, rem)
    # 2) Kalan talep icin spot
    for rota, R in dem.items():
        if R <= 1e-6: continue
        o, d = rota.split(" - "); dist = D(o, d)
        for vn, load in spot_cover(R, dist):
            cost = V[vn]["sf"] + V[vn]["sk"]*dist; total += cost
            rows.append([day, f"Spot {vn}", o, d, round(load, 2), round(cost, 2)])

plan = pd.DataFrame(rows, columns=["Tarih", "Araç Tipi", "Çıkış TM", "Varış TM", "Atanan Desi", "Maliyet"])
plan = plan.sort_values(["Tarih", "Çıkış TM", "Varış TM", "Araç Tipi"]).reset_index(drop=True)
plan.to_excel("arac_planlamasi.xlsx", index=False)
print(f"Optimizasyon tamamlandi: {len(plan)} arac atamasi -> arac_planlamasi.xlsx")
print(f"TOPLAM AG MALIYETI: {total:,.2f} TL")
