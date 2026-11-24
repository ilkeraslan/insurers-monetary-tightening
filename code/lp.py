"""Panel local projections of insurers' net purchases on ECB monetary policy shocks."""
import json
import numpy as np
import pandas as pd

O = "../results/"
P = pd.read_csv(O + "panel.csv", parse_dates=["date"])
P["guar_share"] = P["guar_share"].clip(lower=0, upper=1)
P = P[P["date"] <= "2025-12-31"].copy()
P["mp"] = 10 * P["MP_pm"]      # 10 basis point surprise
P["cbi"] = 10 * P["CBI_pm"]
P["dy10"] = P.groupby("area")["y10"].diff()


def winsor(s, q=0.01):
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


FLOWS = ["ic_f3_s", "ic_f3l_s", "ic_f3s_s", "ic_f52_s", "ic_f5_s", "ic_f2_s", "ic_f6_s", "if_f3_s", "bk_f3_s", "pf_f3_s"]
for c in FLOWS:
    P[c + "_w"] = winsor(P[c])


def dk_se(X, u, t, L):
    """Driscoll-Kraay standard errors: sum scores by period, Bartlett kernel with L lags."""
    XtXi = np.linalg.inv(X.T @ X)
    g = pd.DataFrame(X * u[:, None]).groupby(t.values).sum().sort_index().values
    S = g.T @ g
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    return np.sqrt(np.diag(XtXi @ S @ XtXi))


def panel_lp(df, y, H=8, p=4, inter=None, timefe=False, sample=None, y10=True):
    d = df.copy() if sample is None else df[sample].copy()
    d = d.sort_values(["area", "date"])
    gb = d.groupby("area")
    out = {}
    for h in range(H + 1):
        lhs = sum(gb[y].shift(-k) for k in range(h + 1))
        X = pd.DataFrame(index=d.index)
        X["mp"] = d["mp"]
        if inter is not None:
            z = d[inter] - d.drop_duplicates("area")[inter].mean()
            X["mp_x"] = d["mp"] * z
        X["cbi"] = d["cbi"]
        for i in range(1, p + 1):
            X[f"y{i}"] = gb[y].shift(i)
            X[f"mp{i}"] = gb["mp"].shift(i)
            if y10:
                X[f"dy10_{i}"] = gb["dy10"].shift(i)
        Z = pd.concat([lhs.rename("lhs"), X, d[["area", "date"]]], axis=1).dropna()
        # fixed effects by demeaning (country; and time if requested)
        cols = [c for c in Z.columns if c not in ("area", "date")]
        W = Z[cols] - Z.groupby("area")[cols].transform("mean")
        if timefe:
            W = W - W.groupby(Z["date"])[cols].transform("mean")
            W = W.drop(columns=[c for c in ["mp", "cbi"] + [f"mp{i}" for i in range(1, p + 1)] if c in W])
        W = W.loc[:, (W.abs().sum() > 0) | (W.columns == "lhs")]
        Xm = W.drop(columns="lhs").values
        yv = W["lhs"].values
        b = np.linalg.lstsq(Xm, yv, rcond=None)[0]
        u = yv - Xm @ b
        se = dk_se(Xm, u, Z["date"], h + 4)
        names = list(W.drop(columns="lhs").columns)
        res = {k: (float(b[names.index(k)]), float(se[names.index(k)])) for k in ["mp", "mp_x"] if k in names}
        res["n"] = int(len(Z))
        res["N"] = int(Z["area"].nunique())
        out[h] = res
    return out


if __name__ == "__main__":
    ea = P["ea"] == 1
    R = {}
    for y in FLOWS:
        R[f"base_{y}"] = panel_lp(P, y + "_w", sample=ea)
        R[f"int_{y}"] = panel_lp(P, y + "_w", sample=ea, inter="guar_share", timefe=True)
    R["noneu_ic_f3_s"] = panel_lp(P, "ic_f3_s_w", sample=P["nonea_eu"] == 1, y10=False)
    json.dump(R, open(O + "lp.json", "w"), indent=1)


    def f(v):
        b, se = v
        t = abs(b / se)
        s = "***" if t > 2.58 else "**" if t > 1.96 else "*" if t > 1.645 else ""
        return f"{b:7.3f}({se:.3f}){s}"


    for k, r in R.items():
        key = "mp_x" if k.startswith("int_") else "mp"
        print(f"{k:22s} N={r[0]['N']:2d} n={r[0]['n']:5d} " + " ".join(f(r[h][key]) for h in [0, 1, 2, 4, 6, 8]))
