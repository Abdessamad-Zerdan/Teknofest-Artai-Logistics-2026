# -*- coding: utf-8 -*-
"""
Talep Tahminleme - 11-17 Mayis 2026 (LightGBM kantil regresyon).
Cikti semasi (sartname): Tarih | Çıkış TM | Varış TM | Tahmin Edilen Desi
"""
import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")

df = pd.read_excel("data/Desi_Talep.xlsx")
df["Tarih"] = pd.to_datetime(df["Tarih"])
df["Rota"] = df["Çıkış Transfer Merkezi"] + " - " + df["Varış Transfer Merkezi"]
daily = df.groupby(["Tarih", "Rota"])["Toplam Desi"].sum().reset_index()

future = pd.date_range("2026-05-11", "2026-05-17")
rts = daily["Rota"].unique()
fut = pd.DataFrame([(d, r) for d in future for r in rts], columns=["Tarih", "Rota"]); fut["Toplam Desi"] = np.nan
data = pd.concat([daily, fut], ignore_index=True).sort_values(["Rota", "Tarih"]).reset_index(drop=True)

rm = daily.groupby("Rota")["Toplam Desi"].mean().to_dict()
data["TE"] = data["Rota"].map(rm).fillna(0.0)
data["dow"] = data["Tarih"].dt.dayofweek
data["wknd"] = (data["dow"] >= 5).astype(int)
data["hol"] = ((data["Tarih"].dt.month == 5) & (data["Tarih"].dt.day == 1)).astype(int)
g = data.groupby("Rota")["Toplam Desi"]
data["l7"] = g.shift(7); data["l14"] = g.shift(14)
data["rm7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
data["rs7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
for c in ["l7", "l14", "rm7"]: data[c] = data[c].fillna(data["TE"])
data["rs7"] = data["rs7"].fillna(0.0)

feats = ["dow", "wknd", "hol", "TE", "l7", "l14", "rm7", "rs7"]
tr = data["Tarih"] < "2026-05-11"; te = data["Tarih"] >= "2026-05-11"
m = lgb.LGBMRegressor(objective="quantile", alpha=0.85, n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
m.fit(data.loc[tr, feats], data.loc[tr, "Toplam Desi"])
data.loc[te, "pred"] = np.clip(m.predict(data.loc[te, feats]), 0, None)

out = data.loc[te, ["Tarih", "Rota", "pred"]].copy()
out["Çıkış TM"] = out["Rota"].str.split(" - ").str[0]
out["Varış TM"] = out["Rota"].str.split(" - ").str[1]
out["Tarih"] = out["Tarih"].dt.strftime("%Y-%m-%d")
out = out.rename(columns={"pred": "Tahmin Edilen Desi"})
out = out[["Tarih", "Çıkış TM", "Varış TM", "Tahmin Edilen Desi"]].sort_values(["Tarih", "Çıkış TM", "Varış TM"])
out["Tahmin Edilen Desi"] = out["Tahmin Edilen Desi"].round(2)
out.to_excel("tahmin_talep.xlsx", index=False)
print(f"Tahmin tamamlandi: {len(out)} satir -> tahmin_talep.xlsx")
