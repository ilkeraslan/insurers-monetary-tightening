"""Triple interaction: shock x guaranteed share x moneyness of the surrender option (full sample, country and time FE)."""
import json
import numpy as np
import pandas as pd
from lp import P, dk_se

O = "../results/"
d0 = P[P["ea"] == 1].copy().sort_values(["area", "date"])
g = d0.groupby("area")
# moneyness proxy: current ten-year yield minus its trailing ten-year average (the vintage at which the
# guaranteed book in force was written); positive values mean surrender options are closer to the money
d0["ytrail"] = g["y10"].transform(lambda s: s.shift(1).rolling(40, min_periods=20).mean())
d0["money"] = g["y10"].shift(1) - d0["ytrail"]
G = d0.drop_duplicates("area").set_index("area")["guar_share"].clip(0, 1)
d0["Gd"] = d0["area"].map(G - G.mean())


def triple(df, y, H=8, p=4):
    d = df.copy()
    gb = d.groupby("area")
    out = {}
    for h in range(H + 1):
        Y = sum(gb[y].shift(-k) for k in range(h + 1))
        X = pd.DataFrame(index=d.index)
        X["mpGM"] = d["mp"] * d["Gd"] * d["money"]
        X["mpG"] = d["mp"] * d["Gd"]
        X["mpM"] = d["mp"] * d["money"]
        X["GM"] = d["Gd"] * d["money"]
        X["M"] = d["money"]
        for i in range(1, p + 1):
            X[f"y{i}"] = gb[y].shift(i)
            X[f"dy10_{i}"] = gb["dy10"].shift(i)
        Z = pd.concat([Y.rename("Y"), X, d[["area", "date"]]], axis=1).dropna()
        cols = ["Y"] + list(X.columns)
        W = Z[cols] - Z.groupby("area")[cols].transform("mean")
        W = W - W.groupby(Z["date"])[cols].transform("mean")
        Xm = W[list(X.columns)].values
        b = np.linalg.lstsq(Xm, W["Y"].values, rcond=None)[0]
        u = W["Y"].values - Xm @ b
        se = dk_se(Xm, u, Z["date"], h + 4)
        nm = list(X.columns)
        out[h] = {k: (float(b[nm.index(k)]), float(se[nm.index(k)])) for k in ["mpGM", "mpG", "mpM"]}
        out[h]["n"] = int(len(Z))
    return out


R = {}
for y in ["ic_f6_s_w", "ic_f3_s_w", "ic_f3l_s_w", "ic_f52_s_w"]:
    R[y] = triple(d0, y)
json.dump(R, open(O + "moneyness.json", "w"), indent=1)
print(d0["money"].describe().round(2))
print(d0.groupby(d0["date"].dt.year)["money"].mean().round(2).to_dict())


def f(v):
    b, se = v
    t = abs(b / se)
    s = "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.645 else ""
    return f"{b:7.2f}({se:.2f}){s}"


for y, r in R.items():
    print(y, "n", r[0]["n"])
    for k in ["mpGM", "mpG"]:
        print("  ", k, " ".join(f(r[h][k]) for h in [0, 1, 2, 4, 6, 8]))
