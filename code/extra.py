"""Pre/post-2015 splits, guarantee interactions, break test, sample variants (outputs split.json, extra.json, preint.json)."""
import json
from lp import P, panel_lp
O = "../results/"
ea = P["ea"] == 1
post = ea & (P["date"] >= "2015-01-01")
pre = ea & (P["date"] <= "2014-12-31")
R = {}
for y in ["ic_f3_s", "ic_f3l_s", "ic_f52_s", "ic_f6_s", "ic_f2_s", "if_f3_s", "bk_f3_s"]:
    for nm, sm in [("post", post), ("pre", pre)]:
        R[f"{nm}_{y}"] = panel_lp(P, y + "_w", sample=sm)
for y in ["ic_f3_s", "ic_f6_s", "ic_f52_s"]:
    R[f"postint_{y}"] = panel_lp(P, y + "_w", sample=post, inter="guar_share", timefe=True)
json.dump(R, open(O + "split.json", "w"), indent=1)
X = {}
ex = post & ~P["date"].between("2022-01-01", "2023-12-31")
s2 = ea & (P["date"] >= "2016-01-01")
for y in ["ic_f3_s", "ic_f6_s"]:
    X[f"post_ex2223_{y}"] = panel_lp(P, y + "_w", sample=ex)
    X[f"s2_{y}"] = panel_lp(P, y + "_w", sample=s2)
Q = P.copy()
Q["post"] = (Q["date"] >= "2015-01-01").astype(float)
for y in ["ic_f3_s", "ic_f6_s"]:
    X[f"break_{y}"] = panel_lp(Q, y + "_w", sample=ea, inter="post")
X["noneu_post_ic_f3_s"] = panel_lp(P, "ic_f3_s_w", sample=(P["nonea_eu"] == 1) & (P["date"] >= "2015-01-01"), y10=False)
X["noneu_post_ic_f6_s"] = panel_lp(P, "ic_f6_s_w", sample=(P["nonea_eu"] == 1) & (P["date"] >= "2015-01-01"), y10=False)
T = P.copy(); T["mp"] = T["mp"].clip(lower=0)
X["post_tight_ic_f3_s"] = panel_lp(T, "ic_f3_s_w", sample=post)
E2 = P.copy(); E2["mp"] = E2["mp"].clip(upper=0)
X["post_ease_ic_f3_s"] = panel_lp(E2, "ic_f3_s_w", sample=post)
json.dump(X, open(O + "extra.json", "w"), indent=1)
PI = {}
for y in ["ic_f3_s", "ic_f6_s", "ic_f52_s"]:
    PI[f"preint_{y}"] = panel_lp(P, y + "_w", sample=pre, inter="guar_share", timefe=True)
json.dump(PI, open(O + "preint.json", "w"), indent=1)
