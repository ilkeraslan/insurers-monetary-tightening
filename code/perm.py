"""Randomisation inference: permute the guaranteed share across countries (post-2015 interaction)."""
import json
import numpy as np
import pandas as pd
from lp import P, dk_se

O = "../results/"
rng = np.random.default_rng(42)
post = (P["ea"] == 1) & (P["date"] >= "2015-01-01")
d = P[post].copy().sort_values(["area", "date"])
g = d.groupby("area")
G = d.drop_duplicates("area").set_index("area")["guar_share"].clip(0, 1)
areas = G.index.values


def prep(y, h, p=4):
    Y = sum(g[y].shift(-k) for k in range(h + 1))
    C = pd.DataFrame(index=d.index)
    for i in range(1, p + 1):
        C[f"y{i}"] = g[y].shift(i)
        C[f"dy10_{i}"] = g["dy10"].shift(i)
    Z = pd.concat([Y.rename("Y"), d["mp"].rename("mp"), C, d[["area", "date"]]], axis=1).dropna()
    return Z, list(C.columns)


def coef(Z, cols, Gmap):
    z = Z["area"].map(Gmap - Gmap.mean())
    W = pd.concat([Z["Y"], (Z["mp"] * z).rename("x"), Z[cols]], axis=1)
    W = W - W.groupby(Z["area"]).transform("mean")
    W = W - W.groupby(Z["date"]).transform("mean")
    X = W.drop(columns="Y").values
    return np.linalg.lstsq(X, W["Y"].values, rcond=None)[0][0]


out = {}
for y, h in [("ic_f6_s_w", 4), ("ic_f6_s_w", 6), ("ic_f3_s_w", 8), ("ic_f3l_s_w", 8), ("ic_f52_s_w", 6)]:
    Z, cols = prep(y, h)
    b0 = coef(Z, cols, G)
    draws = []
    for _ in range(1999):
        perm = pd.Series(rng.permutation(G.values), index=areas)
        draws.append(coef(Z, cols, perm))
    draws = np.array(draws)
    p = float((np.sum(np.abs(draws) >= abs(b0)) + 1) / (len(draws) + 1))
    p1 = float((np.sum(draws <= b0) + 1) / (len(draws) + 1))
    out[f"{y}_h{h}"] = {"b": float(b0), "p_two": p, "p_one_neg": p1}
    print(y, h, round(b0, 2), "p two-sided", round(p, 3), "p one-sided", round(p1, 3))
json.dump(out, open(O + "perm.json", "w"), indent=1)
