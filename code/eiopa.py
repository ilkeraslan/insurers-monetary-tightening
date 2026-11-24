"""EIOPA annual Solvency II statistics: surrenders versus premiums by country, 2017-2025."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

D = "../data/"
O = "../results/"
NAME = {"AUSTRIA": "AT", "BELGIUM": "BE", "CROATIA": "HR", "CYPRUS": "CY", "ESTONIA": "EE", "FINLAND": "FI",
        "FRANCE": "FR", "GERMANY": "DE", "GREECE": "GR", "IRELAND": "IE", "ITALY": "IT", "LATVIA": "LV",
        "LITHUANIA": "LT", "LUXEMBOURG": "LU", "MALTA": "MT", "NETHERLANDS": "NL", "PORTUGAL": "PT",
        "SLOVAKIA": "SK", "SLOVENIA": "SI", "SPAIN": "ES"}

a = pd.read_csv(D + "eiopa_annual_pce.csv")
a.columns = ["country", "period", "item", "btype", "code", "value", "extr", "nsub"]
b = pd.read_csv(D + "eiopa_annual_bs.csv")
b.columns = ["country", "period", "utype", "value", "item", "code", "nsub", "extr"]
for x in (a, b):
    x["area"] = x["country"].map(NAME)
    x["year"] = x["period"].str.replace("Y", "").astype(int)
a = a.dropna(subset=["area"]).pivot_table(index=["area", "year"], columns="code", values="value")
b = b.dropna(subset=["area"]).pivot_table(index=["area", "year"], columns="code", values="value")
d = a.join(b).reset_index().sort_values(["area", "year"])
d["tp_life"] = d["R0650"].fillna(0) + d["R0690"].fillna(0)
d["tp_lag"] = d.groupby("area")["tp_life"].shift(1)
d["surr_rate"] = 100 * d["R2700"] / d["tp_lag"]
d["prem_rate"] = 100 * d["R1410"] / d["tp_lag"]
d["claims_rate"] = 100 * d["R1610"] / d["tp_lag"]
d["other_claims_rate"] = d["claims_rate"] - d["surr_rate"]
expo = pd.read_csv(O + "exposure.csv", index_col=0)
d["G"] = d["area"].map(expo["guar_share"].clip(0, 1))
# ten-year yields, annual average
P = pd.read_csv(O + "panel.csv", parse_dates=["date"])
y = P.groupby(["area", P["date"].dt.year])["y10"].mean().rename("y10").reset_index().rename(columns={"date": "year"})
d = d.merge(y, on=["area", "year"], how="left")
d["dy10"] = d.groupby("area")["y10"].diff()
d.to_csv(O + "eiopa_panel.csv", index=False)

res = {}
# 1) aggregate euro area (20 countries) by year, EUR bn
agg = d.groupby("year")[["R1410", "R1610", "R2700", "tp_life"]].sum() / 1000
agg["surr_rate"] = 100 * agg["R2700"] / agg["tp_life"].shift(1)
agg["prem_rate"] = 100 * agg["R1410"] / agg["tp_life"].shift(1)
res["agg"] = agg.round(2).reset_index().to_dict(orient="list")
print(agg.round(2))

# 2) cross-section: change 2021 -> 2023 in surrender and premium rates against the guaranteed share
cs = d[d["year"].isin([2021, 2023])].pivot_table(index="area", columns="year", values=["surr_rate", "prem_rate", "other_claims_rate"])
cs = pd.DataFrame({k: cs[(k, 2023)] - cs[(k, 2021)] for k in ["surr_rate", "prem_rate", "other_claims_rate"]})
cs["G"] = expo["guar_share"].clip(0, 1)
w = d[d["year"] == 2021].set_index("area")["tp_life"]
cs["w"] = w
cs = cs.dropna()
for k in ["surr_rate", "prem_rate", "other_claims_rate"]:
    m = sm.WLS(cs[k], sm.add_constant(cs["G"]), weights=cs["w"]).fit(cov_type="HC1")
    u = sm.OLS(cs[k], sm.add_constant(cs["G"])).fit(cov_type="HC1")
    res[f"cs_{k}"] = {"wls_b": float(m.params["G"]), "wls_se": float(m.bse["G"]), "ols_b": float(u.params["G"]),
                      "ols_se": float(u.bse["G"]), "n": int(m.nobs)}
    print(k, round(m.params["G"], 2), round(m.bse["G"], 2), "| ols", round(u.params["G"], 2), round(u.bse["G"], 2), int(m.nobs))
cs.round(2).to_csv(O + "eiopa_cs.csv")

# 3) annual panel 2018-2025 with country and year FE: rates on dy10 x G
pp = d.dropna(subset=["surr_rate", "prem_rate", "dy10", "G"]).copy()
pp["x"] = pp["dy10"] * (pp["G"] - pp.drop_duplicates("area")["G"].mean())
for k in ["surr_rate", "prem_rate", "other_claims_rate"]:
    Z = pp[["area", "year", k, "x", "dy10"]].dropna()
    W = Z[[k, "x", "dy10"]] - Z.groupby("area")[[k, "x", "dy10"]].transform("mean")
    W = W - W.groupby(Z["year"])[[k, "x", "dy10"]].transform("mean")
    m = sm.OLS(W[k], W[["x", "dy10"]]).fit(cov_type="cluster", cov_kwds={"groups": Z["area"]})
    res[f"panel_{k}"] = {"b_x": float(m.params["x"]), "se_x": float(m.bse["x"]), "b_dy": float(m.params["dy10"]),
                         "se_dy": float(m.bse["dy10"]), "n": int(m.nobs)}
    print("panel", k, "x", round(m.params["x"], 2), round(m.bse["x"], 2), "dy10", round(m.params["dy10"], 2), round(m.bse["dy10"], 2), int(m.nobs))
json.dump(res, open(O + "eiopa.json", "w"), indent=1, default=float)
print(cs.sort_values("G").round(2).to_string())

# ---- link QSA outflows in the 2022-23 tightening to the rise in surrenders and premiums (EIOPA)
P2 = P[(P["ea"] == 1) & P["date"].between("2022-09-30", "2023-12-31")].groupby("area")[["ic_f6", "ic_f3"]].sum()
base = P[(P["ea"] == 1) & (P["date"] == "2022-06-30")].set_index("area")["ic_assets"]
P2 = P2.div(base, axis=0) * 100
s = d.pivot_table(index="area", columns="year", values=["R2700", "R1410", "tp_life"])
dS = (s[("R2700", 2023)] + 0.5 * s[("R2700", 2022)] - 1.5 * s[("R2700", 2021)]) / s[("tp_life", 2021)] * 100
dP = (s[("R1410", 2023)] + 0.5 * s[("R1410", 2022)] - 1.5 * s[("R1410", 2021)]) / s[("tp_life", 2021)] * 100
X = pd.concat([P2, dS.rename("dSurr"), dP.rename("dPrem"), s[("tp_life", 2021)].rename("w")], axis=1).dropna()
X = X[X.index != "MT"]
link = {}
for yv in ["ic_f6", "ic_f3"]:
    for nm, mod in [("wls", sm.WLS(X[yv], sm.add_constant(X[["dSurr", "dPrem"]]), weights=X["w"])),
                    ("ols", sm.OLS(X[yv], sm.add_constant(X[["dSurr", "dPrem"]])))]:
        m = mod.fit(cov_type="HC1")
        link[f"{yv}_{nm}"] = {k: (float(m.params[k]), float(m.bse[k])) for k in ["dSurr", "dPrem"]}
        link[f"{yv}_{nm}"]["n"] = int(m.nobs)
res["link"] = link
json.dump(res, open(O + "eiopa.json", "w"), indent=1, default=float)


def star(b_, se):
    t = abs(b_ / se)
    return "^{***}" if t > 2.58 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


rows = []
for yr in range(2018, 2026):
    r = agg.loc[yr]
    rows.append(f"{yr} & {r['R1410']:.0f} & {r['R2700']:.0f} & {r['prem_rate']:.1f} & {r['surr_rate']:.1f} \\\\")
open(O + "tab_eiopa_agg.tex", "w").write("\n".join(rows))
rows = []
for yv, lab in [("ic_f6", "Technical-reserve flow")]:
    for nm, wl in [("wls", "weighted"), ("ols", "unweighted")]:
        v = link[f"{yv}_{nm}"]
        rows.append(f"{lab}, {wl} & ${v['dSurr'][0]:.2f}{star(*v['dSurr'])}$ & ${v['dPrem'][0]:.2f}{star(*v['dPrem'])}$ & {v['n']} \\\\")
        rows.append(f" & $({v['dSurr'][1]:.2f})$ & $({v['dPrem'][1]:.2f})$ & \\\\[2pt]")
open(O + "tab_eiopa_link.tex", "w").write("\n".join(rows))
print(link)

# Within-country response of the surrender rate to the ten-year yield (country fixed effects)
import statsmodels.formula.api as smf
Zp = pd.read_csv(O + "eiopa_panel.csv").sort_values(["area", "year"])
Zp["dy_l"] = Zp.groupby("area")["dy10"].shift(1)
dp = Zp.dropna(subset=["surr_rate", "y10"]).copy()
dp = dp[dp["area"] != "MT"]
lo_, hi_ = dp["surr_rate"].quantile([0.02, 0.98])
dp["sr"] = dp["surr_rate"].clip(lo_, hi_)
phi = {}
for nm, form, key, dd in [("level", "sr ~ y10 + C(area)", "y10", dp), ("change", "sr ~ dy10 + dy_l + C(area)", "dy_l", dp.dropna(subset=["dy_l"]))]:
    for wn, mod in [("ols", smf.ols(form, data=dd)), ("wls", smf.wls(form, data=dd, weights=dd["tp_lag"]))]:
        m = mod.fit(cov_type="cluster", cov_kwds={"groups": dd["area"]})
        phi[f"{nm}_{wn}"] = [float(m.params[key]), float(m.bse[key]), int(m.nobs), int(dd["area"].nunique())]
json.dump(phi, open(O + "eiopa_phi.json", "w"), indent=1)
print(phi)
