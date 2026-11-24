"""Build the country-quarter panel of insurers' financial transactions and ECB monetary policy shocks."""
import numpy as np
import pandas as pd

D = "../data/"
O = "../results/"

d = pd.read_csv(D + "ecb_insurers.csv")
d["ds"] = d["KEY"].str.split(".").str[0]

# ---------------- QSA: transactions and stocks by sector ----------------
q = d[d["ds"] == "QSA"].copy()
p = q["KEY"].str.split(".", expand=True)
q["area"], q["sec"], q["entry"], q["sto"], q["instr"], q["mat"] = p[3], p[5], p[8], p[9], p[10], p[11]
q["var"] = q["sec"] + "_" + q["entry"] + "_" + q["sto"] + "_" + q["instr"] + "_" + q["mat"]
q["date"] = pd.PeriodIndex(q["TIME"].str.replace("-Q", "Q"), freq="Q").to_timestamp(how="end").normalize()
panel = q.pivot_table(index=["area", "date"], columns="var", values="VALUE").reset_index()
ren = {
    "S128_A_F_F3_T": "ic_f3", "S128_A_F_F3_L": "ic_f3l", "S128_A_F_F3_S": "ic_f3s", "S128_A_F_F2_T": "ic_f2",
    "S128_A_F_F4_T": "ic_f4", "S128_A_F_F5__Z": "ic_f5", "S128_A_F_F51__Z": "ic_f51", "S128_A_F_F52__Z": "ic_f52",
    "S128_L_F_F6__Z": "ic_f6", "S128_A_LE_F__Z": "ic_assets", "S128_A_LE_F3_T": "ic_f3stock",
    "S124_A_F_F3_T": "if_f3", "S124_A_LE_F__Z": "if_assets", "S12K_A_F_F3_T": "bk_f3", "S12K_A_LE_F__Z": "bk_assets",
    "S129_A_F_F3_T": "pf_f3", "S129_A_LE_F__Z": "pf_assets",
}
panel = panel.rename(columns=ren)

# ---------------- ICB: technical reserves by country (Solvency II-based, 2016-) ----------------
ic = d[d["ds"] == "ICB"].copy()
p = ic["KEY"].str.split(".", expand=True)
ic["area"], ic["item"], ic["dtype"], ic["cpa"], ic["cps"] = p[2], p[5], p[7], p[8], p[9]
ic["date"] = pd.PeriodIndex(ic["TIME"].str.replace("-Q", "Q"), freq="Q").to_timestamp(how="end").normalize()
icb_c = ic[(ic["dtype"] == "1") & (ic["area"] != "U2")].pivot_table(index="area", columns="item", values="VALUE",
                                                                      aggfunc=lambda s: s.iloc[:8].mean())
# exposure: share of non-unit-linked life reserves in total technical reserves, average 2016Q1-2017Q4 (pre-determined)
first = ic[(ic["dtype"] == "1") & (ic["area"] != "U2") & (ic["date"] <= "2017-12-31")]
expo = first.pivot_table(index="area", columns="item", values="VALUE", aggfunc="mean")
expo["guar_share"] = expo["L412"] / expo["L40"]
expo["life_share"] = expo["L41"] / expo["L40"]
expo[["guar_share", "life_share"]].to_csv(O + "exposure.csv")
# euro area transactions in debt securities by counterpart sector (2016-)
ea = ic[(ic["area"] == "U2") & (ic["dtype"] == "4")].pivot_table(index="date", columns=["cpa", "cps"], values="VALUE")
ea.columns = [f"ea_{a}_{s}" for a, s in ea.columns]
ea.to_csv(O + "ea_icb_flows.csv")

# ---------------- long-term yields ----------------
irs = d[d["ds"] == "IRS"].copy()
p = irs["KEY"].str.split(".", expand=True)
irs["area"] = p[2]
irs["date"] = pd.PeriodIndex(irs["TIME"], freq="M").to_timestamp(how="end").normalize()
y10 = irs.pivot_table(index="date", columns="area", values="VALUE").resample("QE").mean()
y10l = y10.stack().rename("y10").reset_index().rename(columns={"level_1": "area", "area": "area"})
y10l.columns = ["date", "area", "y10"]
panel = panel.merge(y10l, on=["area", "date"], how="left")
de = y10["DE"].rename("y10_de")
panel = panel.merge(de.reset_index(), on="date", how="left")
panel["spread"] = panel["y10"] - panel["y10_de"]

# ---------------- ECB monetary policy shocks (Jarocinski-Karadi, updated series) ----------------
jk = pd.read_csv(D + "jk.csv")
jk["date"] = pd.to_datetime(dict(year=jk["year"], month=jk["month"], day=1)) + pd.offsets.MonthEnd(0)
jkq = jk.set_index("date")[["MP_pm", "CBI_pm", "MP_median", "CBI_median", "pc1_hf"]].resample("QE").sum()
jkq.to_csv(O + "shocks_q.csv")
panel = panel.merge(jkq.reset_index(), on="date", how="left")

# ---------------- euro area membership ----------------
join = {"AT": 1999, "BE": 1999, "DE": 1999, "ES": 1999, "FI": 1999, "FR": 1999, "IE": 1999, "IT": 1999, "LU": 1999,
        "NL": 1999, "PT": 1999, "GR": 2001, "SI": 2007, "CY": 2008, "MT": 2008, "SK": 2009, "EE": 2011, "LV": 2014,
        "LT": 2015, "HR": 2023, "BG": 2026}
panel["ea"] = [int(a in join and dt.year >= join[a]) for a, dt in zip(panel["area"], panel["date"])]
panel["nonea_eu"] = panel["area"].isin(["DK", "SE", "PL", "CZ", "HU", "RO"]).astype(int)
panel = panel[(panel["ea"] == 1) | (panel["nonea_eu"] == 1)]
panel = panel.merge(expo[["guar_share", "life_share"]].reset_index(), on="area", how="left")

# scale flows by lagged total financial assets of the same sector, percent
panel = panel.sort_values(["area", "date"])
for pre, assets in [("ic", "ic_assets"), ("if", "if_assets"), ("bk", "bk_assets"), ("pf", "pf_assets")]:
    lag = panel.groupby("area")[assets].shift(1)
    for c in [c for c in panel.columns if c.startswith(pre + "_f")]:
        panel[c + "_s"] = 100 * panel[c] / lag
panel.to_csv(O + "panel.csv", index=False)
print(panel.groupby("area").date.agg(["min", "max", "count"]).to_string())
print(panel[["ic_f3_s", "ic_f52_s", "ic_f6_s", "ic_f2_s", "if_f3_s", "MP_pm", "CBI_pm"]].describe().T)
print(expo[["guar_share", "life_share"]].round(2).sort_values("guar_share").T.to_string())
