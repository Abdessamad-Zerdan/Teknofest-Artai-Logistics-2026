# -*- coding: utf-8 -*-
"""
Gelismis Asama - Talep Tahminleme (ARTAI referans cozumu)
Hedef hafta: 29 Haziran - 5 Temmuz 2026 (gecmis veri 28 Haziran'da bitmektedir).
Her rota icin gunde 2 pencere (09:00 / 17:00) ayri tahmin edilir.
Model: LightGBM kantil p50 (medyan) - MVP asamasinda dogrulanan yaklasim.
Cikti: Talep-tahmini.xlsx (resmi sema, D-ID'li)
"""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")

DATA = "data/teknofest26_gelismis.xlsx"
TARGET_DAYS = pd.date_range("2026-06-29", "2026-07-05")

df = pd.read_excel(DATA)
df["tarih"] = pd.to_datetime(df["tarih"])
df["rota"] = df["cikis"] + " | " + df["varis"]
df["win"] = (df["talep_tamamlanma_saati"].astype(str).str.strip() == "17:00").astype(int)

# (tarih, pencere, rota) seviyesine indir
g = df.groupby(["tarih", "win", "rota"])["toplam_desi"].sum().reset_index()

# Anomali temizligi: tatil/ay sonu gunlerinde sistematik ~sifir talep -> ayni haftagunu medyani
g["toplam_desi"] = g["toplam_desi"].astype(float)
day_tot = g.groupby("tarih")["toplam_desi"].sum().reset_index()
day_tot["dow"] = day_tot["tarih"].dt.dayofweek
dow_med = day_tot.groupby("dow")["toplam_desi"].transform("median")
anom_days = set(day_tot.loc[day_tot["toplam_desi"] < 0.25 * dow_med, "tarih"])
print(f"Anomali gun: {len(anom_days)} -> {sorted(str(d.date()) for d in anom_days)}")
g["dow"] = g["tarih"].dt.dayofweek
med = g[~g["tarih"].isin(anom_days)].groupby(["rota", "win", "dow"])["toplam_desi"].median()
mask = g["tarih"].isin(anom_days)
g.loc[mask, "toplam_desi"] = [
    med.get((r, w, d), np.nan) for r, w, d in zip(g.loc[mask, "rota"], g.loc[mask, "win"], g.loc[mask, "dow"])
]
g["toplam_desi"] = g["toplam_desi"].fillna(0)

# Gelecek iskeleti
routes = sorted(g["rota"].unique())
fut = pd.DataFrame([(d, w, r) for d in TARGET_DAYS for w in (0, 1) for r in routes],
                   columns=["tarih", "win", "rota"])
fut["toplam_desi"] = np.nan
allg = pd.concat([g[["tarih", "win", "rota", "toplam_desi"]], fut], ignore_index=True)
allg.sort_values(["rota", "win", "tarih"], inplace=True)
allg.reset_index(drop=True, inplace=True)

# Oznitelikler (rota+pencere serisi bazinda; 7 gunluk tampon -> sizinti yok)
allg["dow"] = allg["tarih"].dt.dayofweek
allg["wknd"] = (allg["dow"] >= 5).astype(int)
train_hist = allg[allg["tarih"] < TARGET_DAYS[0]]
te = train_hist.groupby(["rota", "win"])["toplam_desi"].mean().rename("te").reset_index()
allg = allg.merge(te, on=["rota", "win"], how="left")
allg["te"] = allg["te"].fillna(0)
grp = allg.groupby(["rota", "win"])["toplam_desi"]
for L in (7, 14, 21, 28):
    allg[f"lag{L}"] = grp.shift(L)
allg["rm28"] = grp.transform(lambda s: s.shift(7).rolling(28, min_periods=7).mean())
allg["rs28"] = grp.transform(lambda s: s.shift(7).rolling(28, min_periods=7).std())
for c in ["lag7", "lag14", "lag21", "lag28", "rm28"]:
    allg[c] = allg[c].fillna(allg["te"])
allg["rs28"] = allg["rs28"].fillna(0)

feats = ["dow", "wknd", "win", "te", "lag7", "lag14", "lag21", "lag28", "rm28", "rs28"]
tr = allg["tarih"] < TARGET_DAYS[0]
ts = ~tr
m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, n_estimators=400,
                      learning_rate=0.04, num_leaves=63, min_child_samples=20,
                      colsample_bytree=0.8, random_state=42, verbose=-1)
m.fit(allg.loc[tr, feats], allg.loc[tr, "toplam_desi"])
allg.loc[ts, "pred"] = np.clip(m.predict(allg.loc[ts, feats]), 0, None)

out = allg.loc[ts, ["tarih", "win", "rota", "pred"]].copy()
out[["cikis", "varis"]] = out["rota"].str.split(" \\| ", expand=True)
out["saat"] = np.where(out["win"] == 1, "17:00", "09:00")
out.sort_values(["tarih", "win", "cikis", "varis"], inplace=True)
out.reset_index(drop=True, inplace=True)
out["Talep ID"] = ["D%05d" % (i + 1) for i in range(len(out))]

final = pd.DataFrame({
    "Talep ID": out["Talep ID"],
    "Tarih": out["tarih"].dt.strftime("%d.%m.%Y"),
    "Talep Tamamlama Saati": out["saat"],
    "Çıkış Transfer Merkezi": out["cikis"],
    "Varış Transfer Merkezi": out["varis"],
    "Tahmin Edilen Desi": out["pred"].round(2),
})
final.to_excel("Talep-tahmini.xlsx", index=False)
print(f"Tahmin: {len(final)} satir ({len(routes)} rota x {len(TARGET_DAYS)} gun x 2 pencere)")
print(f"Toplam tahmin desi: {final['Tahmin Edilen Desi'].sum():,.0f}")
