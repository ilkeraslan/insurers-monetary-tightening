"""Robustness checks and descriptive evidence for the insurers paper."""
import json
import numpy as np
import pandas as pd
from lp import P, panel_lp, FLOWS

O = "../results/"
ea = P["ea"] == 1
R = {}
# alternative shock: median-rotation MP shock
Q = P.copy()
Q["mp"] = 10 * Q["MP_median"]
Q["cbi"] = 10 * Q["CBI_median"]
for y in ["ic_f3_s", "ic_f52_s", "ic_f6_s"]:
    R[f"median_{y}"] = panel_lp(Q, y + "_w", sample=Q["ea"] == 1)
# raw high-frequency surprise (no information-effect separation)
Q2 = P.copy()
Q2["mp"] = 10 * Q2["pc1_hf"]
Q2["cbi"] = 0.0
for y in ["ic_f3_s"]:
    R[f"raw_{y}"] = panel_lp(Q2, y + "_w", sample=Q2["ea"] == 1)
# sample splits
pre = ea & (P["date"] <= "2014-12-31")
post = ea & (P["date"] >= "2015-01-01")
nocovid = ea & ~P["date"].between("2020-01-01", "2020-12-31")
for nm, sm in [("pre2015", pre), ("post2015", post), ("nocovid", nocovid)]:
    R[f"{nm}_ic_f3_s"] = panel_lp(P, "ic_f3_s_w", sample=sm)
# unwinsorised
R["raw_w_ic_f3_s"] = panel_lp(P, "ic_f3_s", sample=ea)
# large countries only (DE FR IT NL ES BE AT)
big = ea & P["area"].isin(["DE", "FR", "IT", "NL", "ES", "BE", "AT"])
R["big_ic_f3_s"] = panel_lp(P, "ic_f3_s_w", sample=big)
R["big_ic_f52_s"] = panel_lp(P, "ic_f52_s_w", sample=big)
json.dump(R, open(O + "robust.json", "w"), indent=1)


def f(v):
    b, se = v
    t = abs(b / se)
    s = "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    return f"{b:7.3f}({se:.3f}){s}"


for k, r in R.items():
    print(f"{k:22s} N={r[0]['N']:2d} n={r[0]['n']:5d} " + " ".join(f(r[h]["mp"]) for h in [0, 1, 2, 4, 6, 8]))

# ---- episode 2022Q3-2023Q4: cumulative flows by guarantee exposure (percent of 2022Q2 assets)
E = P[ea & P["date"].between("2022-09-30", "2023-12-31")].copy()
grp = P.drop_duplicates("area").set_index("area")["guar_share"].clip(0, 1)
hi = grp[grp >= grp.median()].index
E["hi"] = E["area"].isin(hi)
base = P[ea & (P["date"] == "2022-06-30")].set_index("area")["ic_assets"]
agg = E.groupby(["hi"])[["ic_f3", "ic_f52", "ic_f5", "ic_f2", "ic_f6"]].sum()
assets = base.groupby(base.index.isin(hi)).sum()
ep = (100 * agg.div(assets.values, axis=0)).round(2)
ep.index = ["Low guarantee share", "High guarantee share"]
ep.to_csv(O + "episode.csv")
print(ep)
print("high-guarantee countries:", sorted(hi))
# euro area aggregate by counterpart (ICB transactions, EUR bn) 2022Q3-2023Q4
eaf = pd.read_csv(O + "ea_icb_flows.csv", parse_dates=["date"]).set_index("date")
cols = [c for c in eaf.columns if c.startswith("ea_W0_")]
print(eaf.loc["2022-09-30":"2023-12-31", cols].sum().round(1))
print(eaf.loc["2016":"2021", cols].sum().round(1) / 6)
