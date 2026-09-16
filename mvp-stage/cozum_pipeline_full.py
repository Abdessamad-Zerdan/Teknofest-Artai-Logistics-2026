# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - Anahat Optimizasyonu / Temel İşlevli Çözüm (MVP)
==================================================================
Bu betik iki teslim dosyasını üretir:
  1) Desi_talep.xlsx              -> "Tahminlenen Talep" (11-17 Mayıs 2026, her O-D hattı x gün)
  2) Araç_Kapasite_Maliyet.xlsx   -> "Araç Planlama" (her satır bir araç ataması) + Özet/Toplam Maliyet

Yöntem (ön değerlendirme raporu ile tutarlı):
  - Talep tahmini: LightGBM Kantil Regresyon (objective=quantile, alpha=0.85)
    -> Asimetrik maliyet yapısı nedeniyle 85. yüzdelik dilim = güvenli üst sınır.
  - Optimizasyon (MVP): Kiralık-öncelikli + Greedy kapasite doldurma (konsolidasyon YOK, uğrama YOK).
  - Mesafe: Haversine (kuş uçuşu). Dönüş rotaları kapsam dışı. Her gün bağımsız.
  - Kısıtlar: Kiralık zorunlu/önce (sabit maliyet, %10 muafiyeti), Spot >= %10 dolu, talep %100 karşılanır.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

# Tahmin kantili ve çıktı klasörü ortam değişkeniyle ayarlanabilir (varsayılan: p85 / Teslim).
#   p85 (güvenli üst sınır)  : ALPHA=0.85 OUT_DIR=Teslim       python3 cozum_pipeline.py
#   p50 (medyan / doğruluk)  : ALPHA=0.50 OUT_DIR=Teslim_p50   python3 cozum_pipeline.py
ALPHA = float(os.environ.get("ALPHA", "0.85"))

BASE = Path(__file__).resolve().parent
SRC  = BASE / "Temel İşlevli Çözüm Dokümanları"
OUT  = BASE / os.environ.get("OUT_DIR", "Teslim")
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1) VERİ YÜKLEME
# ----------------------------------------------------------------------------
demand = pd.read_excel(SRC / "Desi_talep.xlsx")
demand.columns = ["cikis", "varis", "tarih", "desi"]
demand["tarih"] = pd.to_datetime(demand["tarih"])

coord = pd.read_excel(SRC / "Koordinatlar v2.xlsx")
coord.columns = ["tm", "lat", "lon"]
CM = coord.set_index("tm")

vinfo = pd.read_excel(SRC / "Araç_Kapasite_Maliyet.xlsx")
vinfo.columns = ["arac", "kapasite", "kir_kira", "kir_km", "spot_sabit", "spot_km"]
VEH = vinfo.set_index("arac")

kir = pd.read_excel(SRC / "Kiralık_Araçlar.xlsx")
kir.columns = ["cikis", "varis", "adet", "tur"]

HIST_START = demand["tarih"].min()           # 2026-01-01
HIST_END   = demand["tarih"].max()            # 2026-05-10
FC_DAYS    = pd.date_range("2026-05-11", "2026-05-17", freq="D")  # tahmin haftası

# Türkiye 2026 resmi tatilleri (eğitim aralığında geçerli olanlar)
TR_HOLIDAYS = pd.to_datetime([
    "2026-01-01",                                   # Yılbaşı
    "2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22",  # Ramazan Bayramı (arefe + 3 gün)
    "2026-04-23",                                   # Ulusal Egemenlik ve Çocuk Bayramı
    "2026-05-01",                                   # Emek ve Dayanışma Günü
    "2026-05-19",                                   # Atatürk'ü Anma, Gençlik ve Spor Bayramı
])

# ----------------------------------------------------------------------------
# 2) MESAFE (HAVERSINE - kuş uçuşu)
# ----------------------------------------------------------------------------
def haversine(a, b):
    R = 6371.0
    la1, lo1 = np.radians(CM.loc[a, "lat"]), np.radians(CM.loc[a, "lon"])
    la2, lo2 = np.radians(CM.loc[b, "lat"]), np.radians(CM.loc[b, "lon"])
    dla, dlo = la2 - la1, lo2 - lo1
    h = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))

OD_LIST = sorted(demand.groupby(["cikis", "varis"]).groups.keys())
DIST = {od: haversine(od[0], od[1]) for od in OD_LIST}

# ----------------------------------------------------------------------------
# 3) TAM PANEL (eksik O-D/gün kombinasyonları = 0 talep)
# ----------------------------------------------------------------------------
od_df = pd.DataFrame(OD_LIST, columns=["cikis", "varis"])
dates_df = pd.DataFrame({"tarih": pd.date_range(HIST_START, HIST_END, freq="D")})
panel = od_df.merge(dates_df, how="cross")
panel = panel.merge(demand, on=["cikis", "varis", "tarih"], how="left")
panel["desi"] = panel["desi"].fillna(0.0)
panel["od"] = list(zip(panel["cikis"], panel["varis"]))

# ----------------------------------------------------------------------------
# 3.5) ANOMALİ/TATİL TEMİZLİĞİ
#  Veride ay-sonu günleri (31 Oca, 28 Şub, 31 Mar, 30 Nis) ve son gün (10 May)
#  sistematik olarak ~sıfır talep gösteriyor (eksik/bozuk kayıt); ayrıca resmi
#  tatillerde talep çöküyor. Bu günler lag özniteliklerini zehirleyip (örn. 17 May
#  Pazar -> lag_7 = 10 May) tahmini çökertiyor. Bu günleri aynı-haftagünü medyanıyla
#  impute ediyoruz (forecast haftası normal olduğundan lag'ler "normal güne" dayanmalı).
# ----------------------------------------------------------------------------
panel["dow0"] = panel["tarih"].dt.dayofweek
_daily = panel.groupby("tarih")["desi"].sum()
_dow = _daily.index.dayofweek
_wkmed = pd.Series(_daily.values, index=_dow).groupby(level=0).median()
_ratio = _daily / _wkmed.reindex(_dow).values
ABNORMAL_DAYS = set(_daily.index[_ratio < 0.5]) | (set(TR_HOLIDAYS) & set(_daily.index))

panel["desi_raw"] = panel["desi"]
panel["abnormal"] = panel["tarih"].isin(ABNORMAL_DAYS)
_med = panel[~panel["abnormal"]].groupby(["od", "dow0"])["desi"].median().rename("imp")
panel = panel.merge(_med, on=["od", "dow0"], how="left")
panel.loc[panel["abnormal"], "desi"] = panel.loc[panel["abnormal"], "imp"]
panel["desi"] = panel["desi"].fillna(panel["imp"]).fillna(0.0)
panel = panel.drop(columns=["dow0", "imp"])
print(f"Temizlenen anormal/tatil günü: {len(ABNORMAL_DAYS)} -> "
      f"{sorted(str(d.date()) for d in ABNORMAL_DAYS)}")

# ----------------------------------------------------------------------------
# 4) ÖZNİTELİK MÜHENDİSLİĞİ  (DOĞRUDAN / non-recursive 7 günlük blok tahmini)
#    7 gün ileri tahminde tüm öznitelikler GAP=7 gün öncesine dayanır; böylece
#    11-17 Mayıs için hepsi GERÇEK veriden hesaplanır (özyineleme/hata birikimi yok).
# ----------------------------------------------------------------------------
GAP = 7
FEATURES = ["cikis", "varis", "dow", "month", "weekofyear", "quarter",
            "is_weekend", "is_holiday",
            "lag_7", "lag_14", "lag_21", "lag_28",
            "dow_mean4", "dow_max4",
            "roll4w_mean", "roll4w_std", "roll4w_max",
            "roll8w_mean", "roll8w_std", "roll8w_max",
            "origin_total_lag7", "dest_total_lag7"]
CAT_FEATURES = ["cikis", "varis"]

def add_calendar(df):
    t = df["tarih"]
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["weekofyear"] = t.dt.isocalendar().week.astype(int)
    df["quarter"] = t.dt.quarter
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = t.isin(TR_HOLIDAYS).astype(int)
    return df

def build_features(work):
    """work: od-bazlı günlük panel (tarih sıralı). 7 gün gecikmeli, sızıntısız öznitelikler."""
    work = work.sort_values(["od", "tarih"]).copy()
    g = work.groupby("od")["desi"]
    # Aynı haftagünü geçmişi (mevsimsel sinyalin kalbi)
    for L in (7, 14, 21, 28):
        work[f"lag_{L}"] = g.shift(L)
    work["dow_mean4"] = work[[f"lag_{L}" for L in (7, 14, 21, 28)]].mean(axis=1)
    work["dow_max4"]  = work[[f"lag_{L}" for L in (7, 14, 21, 28)]].max(axis=1)
    # Hareketli istatistikler (GAP=7 kaydırmalı -> 7 gün ileride bile erişilebilir)
    sh = g.shift(GAP)
    for W in (4, 8):  # son 4 ve 8 hafta
        roll = sh.groupby(work["od"]).rolling(W * 7, min_periods=7)
        work[f"roll{W}w_mean"] = roll.mean().reset_index(level=0, drop=True)
        work[f"roll{W}w_std"]  = roll.std().reset_index(level=0, drop=True)
        work[f"roll{W}w_max"]  = roll.max().reset_index(level=0, drop=True)
    # Uzamsal gecikme: çıkış/varış TM'sinin 7 gün önceki toplam talebi (ağ-seviyesi sinyal)
    do = work.groupby(["cikis", "tarih"])["desi"].sum().rename("o_tot").reset_index()
    do["o_lag7"] = do.groupby("cikis")["o_tot"].shift(GAP)
    dd = work.groupby(["varis", "tarih"])["desi"].sum().rename("d_tot").reset_index()
    dd["d_lag7"] = dd.groupby("varis")["d_tot"].shift(GAP)
    work = work.merge(do[["cikis", "tarih", "o_lag7"]], on=["cikis", "tarih"], how="left")
    work = work.merge(dd[["varis", "tarih", "d_lag7"]], on=["varis", "tarih"], how="left")
    work = work.rename(columns={"o_lag7": "origin_total_lag7", "d_lag7": "dest_total_lag7"})
    work = add_calendar(work)
    for c in CAT_FEATURES:
        work[c] = work[c].astype("category")
    return work

LGB_PARAMS = dict(objective="quantile", alpha=ALPHA, n_estimators=700,
                  learning_rate=0.05, num_leaves=63, min_child_samples=30,
                  subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                  reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)

def train_model(feat_table):
    feat = feat_table.dropna(subset=["lag_28", "roll4w_mean"])
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(feat[FEATURES], feat["desi"], categorical_feature=CAT_FEATURES)
    return model

# ----------------------------------------------------------------------------
# 5) WALK-FORWARD DOĞRULAMA (son 5 hafta) - dürüst kalibrasyon raporu
#    Her hafta yalnızca o haftadan önceki veriyle eğitilip tahmin edilir.
# ----------------------------------------------------------------------------
def pinball(y, yhat, a=ALPHA):
    d = y - yhat
    return np.mean(np.maximum(a * d, (a - 1) * d))

feat_all = build_features(panel)  # tüm panel için öznitelikler (bir kez)

VAL_STARTS = ["2026-04-06", "2026-04-13", "2026-04-20", "2026-04-27", "2026-05-04"]
ys, ps = [], []
for s in VAL_STARTS:
    vdays = pd.date_range(s, periods=7, freq="D")
    tr = feat_all[feat_all["tarih"] < vdays[0]]
    te = feat_all[feat_all["tarih"].isin(vdays) & ~feat_all["abnormal"]]  # anormal günler hariç
    mdl = train_model(tr)
    p = np.clip(mdl.predict(te[FEATURES]), 0, None)
    ys.append(te["desi_raw"].values); ps.append(p)  # gerçek (ham) talebe karşı
yv, pv = np.concatenate(ys), np.concatenate(ps)
val_metrics = dict(
    MAE=float(np.mean(np.abs(yv - pv))),
    RMSE=float(np.sqrt(np.mean((yv - pv) ** 2))),
    Pinball=float(pinball(yv, pv, ALPHA)),
    Coverage=float(np.mean(yv <= pv)),   # gerçeklesenin tahminin altinda kalma orani
    n=int(len(yv)))
print(f"Walk-forward (5 hafta, p{int(ALPHA*100)}):", {k: round(v, 3) for k, v in val_metrics.items()})

# ----------------------------------------------------------------------------
# 6) NİHAİ MODEL + 11-17 MAYIS TAHMİNİ  (doğrudan; lag>=7 hepsi gerçek veriden)
# ----------------------------------------------------------------------------
model_final = train_model(feat_all)

# Tahmin haftasını panele ekle (desi=NaN), öznitelikleri yeniden hesapla, doğrudan tahmin et
fc_rows = od_df.merge(pd.DataFrame({"tarih": FC_DAYS}), how="cross")
fc_rows["desi"] = np.nan
fc_rows["od"] = list(zip(fc_rows["cikis"], fc_rows["varis"]))
panel_ext = pd.concat([panel, fc_rows], ignore_index=True)
feat_ext = build_features(panel_ext)
fc_feat = feat_ext[feat_ext["tarih"].isin(FC_DAYS)].copy()
fc_feat["pred"] = np.clip(model_final.predict(fc_feat[FEATURES]), 0, None)
forecast = fc_feat[["cikis", "varis", "tarih", "pred"]].rename(columns={"pred": "desi"})
forecast["desi"] = forecast["desi"].round(2)
forecast = forecast.sort_values(["tarih", "cikis", "varis"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 7) ARAÇ PLANLAMA (Kiralık-öncelikli + Greedy Spot + Uğrama ile tiny-load birleştirme)
# ----------------------------------------------------------------------------
TYPES = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]
CAP = {t: float(VEH.loc[t, "kapasite"]) for t in TYPES}
MIN_FILL = 0.10
SMALL = MIN_FILL * CAP["Kamyonet"]   # 560 desi: en küçük aracın %10'u

def spot_cost(t, dist):  return float(VEH.loc[t, "spot_sabit"] + VEH.loc[t, "spot_km"] * dist)
def rented_cost(t, dist): return float(VEH.loc[t, "kir_kira"] + VEH.loc[t, "kir_km"] * dist)

def distribute_load(total, caps):
    """total yükü kapasiteye orantılı dağıtır (kapasiteyi aşmadan)."""
    caps = np.array(caps, float)
    if total <= 0 or len(caps) == 0:
        return np.zeros(len(caps))
    load = np.minimum(caps, total * caps / caps.sum())
    leftover = total - load.sum()
    while leftover > 1e-6:
        free = caps - load
        idx = np.where(free > 1e-6)[0]
        if len(idx) == 0: break
        add = min(leftover, free[idx].sum())
        load[idx] += add * free[idx] / free[idx].sum()
        leftover = total - load.sum()
    return load

def nn_route(origin, dests):
    """Çıkıştan başlayıp en yakın komşu sırasıyla durakları ziyaret eder (uğrama güzergâhı)."""
    route, cur, rem = [], origin, list(dests)
    while rem:
        nxt = min(rem, key=lambda d: haversine(cur, d))
        route.append(nxt); rem.remove(nxt); cur = nxt
    return route

kir_map = {}
for _, r in kir.iterrows():
    kir_map.setdefault((r["cikis"], r["varis"]), []).append((r["tur"], int(r["adet"])))

plan_rows = []
sefer = 0                 # benzersiz fiziksel araç/sefer numarası
tiny_pool = {}            # (gün, çıkış TM) -> [(varış TM, yük), ...]  (uğrama ile birleştirilecek)

def add_row(day, c, v, tur, tip, dist, load, fixed, kmcost, fill_pct, note=""):
    plan_rows.append({
        "Tarih": day, "Çıkış Transfer Merkezi": c, "Varış Transfer Merkezi": v,
        "Araç Türü": tur, "Araç Tipi": tip, "Sefer No": sefer,
        "Mesafe (km)": round(dist, 2), "Kapasite (desi)": CAP[tur],
        "Yüklenen Desi": round(load, 2), "Doluluk Oranı (%)": round(fill_pct, 1),
        "Günlük Sabit/Kira (TL)": round(fixed, 2), "Mesafe Maliyeti (TL)": round(kmcost, 2),
        "Toplam Maliyet (TL)": round(fixed + kmcost, 2), "Not": note})

for _, r in forecast.iterrows():
    od = (r["cikis"], r["varis"]); c, v = od; dist = DIST[od]; D = float(r["desi"]); day = r["tarih"]

    # --- Kiralık (zorunlu, önce yüklenir, %10 muaf, kullanılsa da kullanılmasa da gider) ---
    rented = [t for tur, adet in kir_map.get(od, []) for t in [tur] * adet]
    rented_cap = sum(CAP[t] for t in rented)
    rented_loads = distribute_load(min(D, rented_cap), [CAP[t] for t in rented])
    for i, t in enumerate(rented):
        sefer += 1
        add_row(day, c, v, t, "Kiralık", dist, rented_loads[i],
                float(VEH.loc[t, "kir_kira"]), float(VEH.loc[t, "kir_km"] * dist),
                100 * rented_loads[i] / CAP[t])

    # --- Spot (kalan talep) ---
    rem = max(0.0, D - rented_cap)
    if rem <= 1e-6:
        continue
    n_tir = int(rem // CAP["Tır"]); rem2 = rem - n_tir * CAP["Tır"]
    reg = ["Tır"] * n_tir
    tiny = 0.0
    if rem2 > 1e-6:
        opts = [(spot_cost(t, dist), t) for t in TYPES
                if CAP[t] >= rem2 - 1e-6 and rem2 >= MIN_FILL * CAP[t] - 1e-6]
        if opts:
            reg.append(min(opts)[1])           # rem2 tek araçla >=%10 kapanıyor
        elif n_tir >= 1:
            reg.append("Kamyonet")             # Tır'larla dengelenince Kamyonet >=%10 dolar
        else:
            tiny = rem2                         # bütün O-D talebi < 560: uğrama havuzuna at
    if reg:
        loads = distribute_load(rem - tiny, [CAP[t] for t in reg])
        for i, t in enumerate(reg):
            sefer += 1
            add_row(day, c, v, t, "Spot", dist, loads[i],
                    float(VEH.loc[t, "spot_sabit"]), float(VEH.loc[t, "spot_km"] * dist),
                    100 * loads[i] / CAP[t])
    if tiny > 0:
        tiny_pool.setdefault((day, c), []).append((v, tiny))

# --- Uğrama: aynı çıkış TM'sinden giden küçük yükleri tek Kamyonet çok-duraklı seferde birleştir ---
ug_trips = 0; residual_under10 = 0
for (day, origin), items in tiny_pool.items():
    load_map = {v: l for v, l in items}
    order = nn_route(origin, list(load_map.keys()))
    # NN sırasında Kamyonet kapasitesine (5600) göre kutula
    bins, cur, tot = [], [], 0.0
    for dest in order:
        l = load_map[dest]
        if tot + l > CAP["Kamyonet"] + 1e-6 and cur:
            bins.append(cur); cur, tot = [], 0.0
        cur.append(dest); tot += l
    if cur: bins.append(cur)
    for b in bins:
        sefer += 1; ug_trips += 1
        trip_total = sum(load_map[d] for d in b)
        fill = 100 * trip_total / CAP["Kamyonet"]
        if trip_total < SMALL - 1e-6:
            residual_under10 += 1
        prev = origin
        for j, dest in enumerate(b):
            leg = haversine(prev, dest)
            fixed = float(VEH.loc["Kamyonet", "spot_sabit"]) if j == 0 else 0.0
            note = "Uğrama (çok-duraklı sefer)" + (" - asgari taşıma (<%10)" if trip_total < SMALL else "")
            add_row(day, origin, dest, "Kamyonet", "Spot", leg, load_map[dest],
                    fixed, float(VEH.loc["Kamyonet", "spot_km"] * leg), fill, note)
            prev = dest

plan = pd.DataFrame(plan_rows)
COLS = ["Tarih", "Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "Araç Türü", "Araç Tipi",
        "Sefer No", "Mesafe (km)", "Kapasite (desi)", "Yüklenen Desi", "Doluluk Oranı (%)",
        "Günlük Sabit/Kira (TL)", "Mesafe Maliyeti (TL)", "Toplam Maliyet (TL)", "Not"]
plan = plan[COLS].sort_values(["Tarih", "Çıkış Transfer Merkezi", "Varış Transfer Merkezi",
                               "Araç Tipi", "Sefer No"]).reset_index(drop=True)

# ----------------------------------------------------------------------------
# 8) MALİYET ÖZETİ + KONTROLLER
# ----------------------------------------------------------------------------
kir_total  = plan.loc[plan["Araç Tipi"] == "Kiralık", "Toplam Maliyet (TL)"].sum()
spot_total = plan.loc[plan["Araç Tipi"] == "Spot", "Toplam Maliyet (TL)"].sum()
grand = kir_total + spot_total
n_kir_veh  = plan.loc[plan["Araç Tipi"] == "Kiralık", "Sefer No"].nunique()
n_spot_veh = plan.loc[plan["Araç Tipi"] == "Spot", "Sefer No"].nunique()

chk_demand = forecast["desi"].sum()
served = plan["Yüklenen Desi"].sum()
# araç bazında doluluk (sefer başına tek satır) -> %10 altı spot araç sayısı
veh_fill = plan[plan["Araç Tipi"] == "Spot"].groupby("Sefer No")["Doluluk Oranı (%)"].first()
under10 = int((veh_fill < 10 - 1e-6).sum())
print(f"\nTahmin toplam desi: {chk_demand:,.0f} | Plana yüklenen: {served:,.0f} | "
      f"Fark: {chk_demand - served:,.2f}")
print(f"Kiralık araç: {n_kir_veh} | Spot araç: {n_spot_veh} | Uğrama seferi: {ug_trips} | "
      f"%10 altı spot araç: {under10}")
print(f"Kiralık toplam: {kir_total:,.0f} TL | Spot toplam: {spot_total:,.0f} TL | "
      f"GENEL TOPLAM: {grand:,.0f} TL")

# maliyet kırılımı (araç türü x tip)
brk = (plan.groupby(["Araç Tipi", "Araç Türü"])
           .agg(Arac=("Sefer No", "nunique"), Maliyet=("Toplam Maliyet (TL)", "sum"))
           .reset_index())

ozet = pd.DataFrame({
    "Kalem": ["Tahmin Haftası", "Toplam Tahmin Edilen Desi", "Karşılanan Desi",
              "Talep Karşılama Oranı (%)",
              "Kiralık Araç Sefer Sayısı", "Spot Araç Sefer Sayısı", "Uğrama (çok-duraklı) Sefer",
              "Kiralık Toplam Maliyet (TL)", "Spot Toplam Maliyet (TL)", "GENEL TOPLAM MALİYET (TL)",
              "Tahmin Modeli", "Tahmin Kantili (alpha)", "Mesafe Yöntemi",
              "Validasyon - MAE (desi)", "Validasyon - RMSE (desi)",
              f"Validasyon - Pinball@{ALPHA}", f"Validasyon - Coverage (p{int(ALPHA*100)})",
              "Min. doluluk (<%10) zorunlu kalan araç"],
    "Değer": ["11-17 Mayıs 2026", round(chk_demand, 2), round(served, 2),
              round(100 * served / chk_demand, 2),
              n_kir_veh, n_spot_veh, ug_trips,
              round(kir_total, 2), round(spot_total, 2), round(grand, 2),
              "LightGBM Kantil Regresyon (objective=quantile)", ALPHA, "Haversine (kuş uçuşu)",
              round(val_metrics["MAE"], 1), round(val_metrics["RMSE"], 1),
              round(val_metrics["Pinball"], 1), round(val_metrics["Coverage"], 3),
              under10]})

_alpha_note = ("asimetrik maliyet -> güvenli üst sınır / kapasite tamponu" if ALPHA >= 0.6
               else "medyan tahmini -> nokta-tahmin doğruluğu (MAE) odaklı, kapasite tamponu yok"
               if abs(ALPHA - 0.5) < 1e-9 else f"{int(ALPHA*100)}. yüzdelik dilim")
varsayim = pd.DataFrame({"Yöntem ve Varsayımlar": [
    "1) Tahmin haftası 11-17 Mayıs 2026; geçmiş veri 1 Ocak-10 Mayıs 2026 (130 gün, 89 O-D hattı).",
    f"2) Talep tahmini: LightGBM kantil regresyon, alpha={ALPHA} ({_alpha_note}).",
    "   Öznitelikler: aynı-haftagünü lag (7/14/21/28), 4/8 haftalık hareketli ist., takvim, tatil, uzamsal lag.",
    "   Doğrulama: 5 haftalık walk-forward (ileriye dönük), anomali/tatil günleri hariç.",
    "3) Veri temizliği: ay-sonu (31 Oca/28 Şub/31 Mar/30 Nis) ve son gün (10 May) sistematik ~sıfır",
    "   anomalileri + resmi tatiller, lag'leri bozmamak için aynı-haftagünü medyanıyla impute edildi.",
    "4) Mesafe: Haversine (kuş uçuşu). Araç dönüş/boş dönüş maliyetleri kapsam dışı. Her gün bağımsız.",
    "5) Kiralık araçlar veri setinde tanımlı hat/tip/sayı ile sabittir; önce ve zorunlu kullanılır,",
    "   %10 doluluk kuralından muaftır, kullanılsın/kullanılmasın maliyeti (kira + km) yansır.",
    "6) Spot araç havuzu her TM'de sınırsızdır; yalnızca kiralık kapasiteyi aşan talep için eklenir,",
    "   her spot araç >= %10 dolu olmalıdır. Tır birim-desi maliyetinde baskın -> yığın Tır + kalan en ucuz araç.",
    "7) Konsolidasyon YAPILMAZ (MVP). Uğrama (çok-duraklı sefer) serbesttir: aynı çıkış TM'sinden",
    "   giden 560 desi altı küçük yükler, her araç >=%10 dolacak şekilde tek Kamyonet seferinde birleştirilir.",
    "8) Maliyet = Günlük sabit/kira + (km başına maliyet x mesafe); hem kiralık hem spot için ikisi de dahil.",
    "9) Talep %100 karşılanır (katı kısıt). Bir çıkış TM'sinin tüm günlük küçük talebi 560 desi altındaysa",
    "   ve birleştirilecek başka yük yoksa, zorunlu taşıma için tek Kamyonet gönderilir (kaçınılmaz köşe durum).",
]})

# ----------------------------------------------------------------------------
# 9) TESLİM DOSYALARINI YAZ + BİÇİMLENDİR
# ----------------------------------------------------------------------------
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

def fmt_sheet(ws, money_cols=(), int_cols=(), wrap_cols=()):
    head_fill = PatternFill("solid", fgColor="1F4E78"); head_font = Font(bold=True, color="FFFFFF")
    for cell in ws[1]:
        cell.fill = head_fill; cell.font = head_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[letter].width = min(max(width + 2, 11), 48)
        header = ws.cell(row=1, column=col[0].column).value
        for c in col[1:]:
            if header in money_cols:  c.number_format = "#,##0.00"
            elif header in int_cols:  c.number_format = "#,##0"
            if header in wrap_cols:   c.alignment = Alignment(wrap_text=True, vertical="top")

demand_out = (forecast.rename(columns={"cikis": "Çıkış Transfer Merkezi", "varis": "Varış Transfer Merkezi",
                                       "tarih": "Tarih", "desi": "Toplam Desi"}))
demand_out["Tarih"] = pd.to_datetime(demand_out["Tarih"]).dt.strftime("%Y-%m-%d")
demand_out = demand_out[["Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "Tarih", "Toplam Desi"]]
with pd.ExcelWriter(OUT / "Desi_talep.xlsx", engine="openpyxl") as xw:
    demand_out.to_excel(xw, index=False, sheet_name="Tahminlenen Talep")
    fmt_sheet(xw.sheets["Tahminlenen Talep"], money_cols=("Toplam Desi",))

plan_w = plan.copy()
plan_w["Tarih"] = pd.to_datetime(plan_w["Tarih"]).dt.strftime("%Y-%m-%d")
with pd.ExcelWriter(OUT / "Araç_Kapasite_Maliyet.xlsx", engine="openpyxl") as xw:
    plan_w.to_excel(xw, index=False, sheet_name="Araç Planlama")
    ozet.to_excel(xw, index=False, sheet_name="Özet")
    brk.to_excel(xw, index=False, sheet_name="Maliyet Kırılımı")
    varsayim.to_excel(xw, index=False, sheet_name="Yöntem ve Varsayımlar")
    fmt_sheet(xw.sheets["Araç Planlama"],
              money_cols=("Yüklenen Desi", "Kapasite (desi)", "Günlük Sabit/Kira (TL)",
                          "Mesafe Maliyeti (TL)", "Toplam Maliyet (TL)", "Mesafe (km)"),
              int_cols=("Sefer No",), wrap_cols=("Not",))
    fmt_sheet(xw.sheets["Özet"], wrap_cols=("Değer",))
    fmt_sheet(xw.sheets["Maliyet Kırılımı"], money_cols=("Maliyet",), int_cols=("Arac",))
    fmt_sheet(xw.sheets["Yöntem ve Varsayımlar"], wrap_cols=("Yöntem ve Varsayımlar",))

print(f"\nYazıldı:\n  {OUT/'Desi_talep.xlsx'}  ({len(demand_out)} satır)")
print(f"  {OUT/'Araç_Kapasite_Maliyet.xlsx'}  ({len(plan_w)} satır, {n_kir_veh+n_spot_veh} araç)")
