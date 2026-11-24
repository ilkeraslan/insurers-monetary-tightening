"""Mechanism quantification (LP-IV), sovereign-spread response, leave-one-country-out."""
import json
import numpy as np
import pandas as pd
from lp import P, panel_lp, dk_se

O = "../results/"
ea = P["ea"] == 1
post = ea & (P["date"] >= "2015-01-01")
pre = ea & (P["date"] <= "2014-12-31")


def fe_demean(Z, cols, timefe=True):
    W = Z[cols] - Z.groupby("area")[cols].transform("mean")
    if timefe:
        W = W - W.groupby(Z["date"])[cols].transform("mean")
    return W


def lp_iv(df, ynum, xend, H=8, p=4, sample=None):
    """Cumulative bond purchases on cumulative technical-reserve flows, instrumented by mp x G; country and time FE."""
    d = df[sample].copy().sort_values(["area", "date"])
    gb = d.groupby("area")
    z = d["guar_share"] - d.drop_duplicates("area")["guar_share"].mean()
    out = {}
    for h in range(H + 1):
        Y = sum(gb[ynum].shift(-k) for k in range(h + 1))
        Xe = sum(gb[xend].shift(-k) for k in range(h + 1))
        C = pd.DataFrame(index=d.index)
        for i in range(1, p + 1):
            C[f"y{i}"] = gb[ynum].shift(i)
            C[f"x{i}"] = gb[xend].shift(i)
            C[f"dy10_{i}"] = gb["dy10"].shift(i)
        Z = pd.concat([Y.rename("Y"), Xe.rename("X"), (d["mp"] * z).rename("Zi"), C, d[["area", "date"]]], axis=1).dropna()
        cols = ["Y", "X", "Zi"] + list(C.columns)
        W = fe_demean(Z, cols)
        Cm = W[list(C.columns)].values
        # first stage
        F = np.column_stack([W["Zi"].values, Cm])
        g = np.linalg.lstsq(F, W["X"].values, rcond=None)[0]
        xhat = F @ g
        ufs = W["X"].values - xhat
        se_fs = dk_se(F, ufs, Z["date"], h + 4)
        Fstat = (g[0] / se_fs[0]) ** 2
        S2 = np.column_stack([xhat, Cm])
        b = np.linalg.lstsq(S2, W["Y"].values, rcond=None)[0]
        u = W["Y"].values - np.column_stack([W["X"].values, Cm]) @ b
        se = dk_se(S2, u, Z["date"], h + 4)
        out[h] = {"b": float(b[0]), "se": float(se[0]), "F": float(Fstat), "n": int(len(Z))}
    return out


R = {}
R["iv_post"] = lp_iv(P, "ic_f3_s_w", "ic_f6_s_w", sample=post)
R["iv_post_long"] = lp_iv(P, "ic_f3l_s_w", "ic_f6_s_w", sample=post)

# sovereign spreads: cumulative change in 10y spread to Bund, interaction with guarantee share, time and country FE
Q = P[ea & (P["area"] != "DE")].copy().sort_values(["area", "date"])
gq = Q.groupby("area")
for h in range(9):
    Q[f"dsp{h}"] = gq["spread"].shift(-h) - gq["spread"].shift(1)
Q["dsp"] = gq["spread"].diff()
sp = {}
for nm, sm in [("pre", Q["date"] <= "2014-12-31"), ("post", Q["date"] >= "2015-01-01")]:
    d = Q[sm].copy()
    z = d["guar_share"] - d.drop_duplicates("area")["guar_share"].mean()
    res = {}
    for h in range(9):
        C = pd.DataFrame({"mpx": d["mp"] * z}, index=d.index)
        g2 = d.groupby("area")
        for i in range(1, 5):
            C[f"l{i}"] = g2["dsp"].shift(i)
        Z = pd.concat([d[f"dsp{h}"].rename("Y"), C, d[["area", "date"]]], axis=1).dropna()
        cols = ["Y"] + list(C.columns)
        W = fe_demean(Z, cols)
        Xm = W[list(C.columns)].values
        b = np.linalg.lstsq(Xm, W["Y"].values, rcond=None)[0]
        u = W["Y"].values - Xm @ b
        se = dk_se(Xm, u, Z["date"], h + 4)
        res[h] = (float(b[0]) * 100, float(se[0]) * 100, int(len(Z)))   # basis points
    sp[nm] = res
R["spread"] = sp

# leave-one-country-out for the post-2015 technical-reserve and debt-security interactions at h=6
loo = {}
for c in sorted(P.loc[post, "area"].unique()):
    s = post & (P["area"] != c)
    r6 = panel_lp(P, "ic_f6_s_w", H=6, sample=s, inter="guar_share", timefe=True)[6]["mp_x"]
    r8 = panel_lp(P, "ic_f3_s_w", H=8, sample=s, inter="guar_share", timefe=True)[8]["mp_x"]
    loo[c] = {"f6_h6": r6, "f3_h8": r8}
R["loo"] = loo
json.dump(R, open(O + "mech.json", "w"), indent=1)

for k in ["iv_post", "iv_post_long"]:
    print(k, [(h, round(R[k][h]["b"], 2), round(R[k][h]["se"], 2), round(R[k][h]["F"], 1)) for h in [2, 4, 6, 8]])
for nm in ["pre", "post"]:
    print("spread", nm, [(h, round(sp[nm][h][0], 1), round(sp[nm][h][1], 1)) for h in [0, 1, 2, 4, 6, 8]])
f6 = [v["f6_h6"][0] for v in loo.values()]
f3 = [v["f3_h8"][0] for v in loo.values()]
print("LOO f6 h6 range", round(min(f6), 2), round(max(f6), 2), "f3 h8 range", round(min(f3), 2), round(max(f3), 2))
print({k: (round(v["f3_h8"][0], 2), round(v["f3_h8"][1], 2)) for k, v in loo.items()})

# ---- tables
def star(b, se):
    t = abs(b / se)
    return "^{***}" if t > 2.58 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""


L = []
for k, nm in [("iv_post", "Debt securities, all maturities"), ("iv_post_long", "Debt securities, long-term")]:
    L.append(nm + " & " + " & ".join(f"${R[k][h]['b']:.2f}{star(R[k][h]['b'], R[k][h]['se'])}$" for h in [2, 4, 6, 8]) + " \\\\")
    L.append(" & " + " & ".join(f"$({R[k][h]['se']:.2f})$" for h in [2, 4, 6, 8]) + " \\\\")
    L.append("\\quad first-stage $F$ & " + " & ".join(f"{R[k][h]['F']:.1f}" for h in [2, 4, 6, 8]) + " \\\\[3pt]")
open(O + "tab_iv.tex", "w").write("\n".join(L))
L = []
for nm, lab in [("pre", "1999--2014"), ("post", "2015--2025")]:
    v = sp[nm]
    L.append(lab + " & " + " & ".join(f"${v[h][0]:.1f}{star(v[h][0], v[h][1])}$" for h in [0, 1, 2, 4, 6, 8]) + " \\\\")
    L.append(" & " + " & ".join(f"$({v[h][1]:.1f})$" for h in [0, 1, 2, 4, 6, 8]) + " \\\\[2pt]")
open(O + "tab_spread.tex", "w").write("\n".join(L))
