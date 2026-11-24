"""Figures and LaTeX tables for the insurers paper."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lp import P, panel_lp

O = "../results/"
ea = P["ea"] == 1
pre = ea & (P["date"] <= "2014-12-31")
post = ea & (P["date"] >= "2015-01-01")
S = json.load(open(O + "split.json"))
B = json.load(open(O + "lp.json"))
X = json.load(open(O + "extra.json"))
RB = json.load(open(O + "robust.json"))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def irf(ax, r, key="mp", title="", ls="-", label=None, band=True):
    hs = list(range(9))
    b = np.array([r[str(h)][key][0] for h in hs])
    se = np.array([r[str(h)][key][1] for h in hs])
    if band:
        ax.fill_between(hs, b - 1.645 * se, b + 1.645 * se, color="0.85" if ls == "-" else "0.93")
    ax.plot(hs, b, color="0.15", ls=ls, lw=1.4, label=label)
    ax.axhline(0, color="0.4", lw=0.6)
    if title:
        ax.set_title(title, fontsize=8.5)


# Figure 1: pre vs post 2015, debt securities, technical reserves, fund shares
fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
for j, (y, t) in enumerate([("ic_f3_s", "Net purchases of debt securities"),
                            ("ic_f6_s", "Net technical-reserve inflows"),
                            ("ic_f52_s", "Net purchases of fund shares")]):
    irf(axes[0, j], S[f"pre_{y}"], title=t)
    irf(axes[1, j], S[f"post_{y}"])
axes[0, 0].set_ylabel("1999–2014\n% of financial assets")
axes[1, 0].set_ylabel("2015–2025\n% of financial assets")
for ax in axes[1]:
    ax.set_xlabel("Quarters after shock")
fig.tight_layout()
fig.savefig(O + "fig_prepost.pdf")

# Figure 2: post-2015 interaction with guarantee share
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharex=True)
for ax, (y, t) in zip(axes, [("ic_f3_s", "Debt securities"), ("ic_f6_s", "Technical reserves"),
                             ("ic_f52_s", "Fund shares")]):
    irf(ax, S[f"postint_{y}"], key="mp_x", title=t)
    ax.set_xlabel("Quarters after shock")
axes[0].set_ylabel("Differential response per\nunit of guarantee share, pp")
fig.tight_layout()
fig.savefig(O + "fig_guarantee.pdf")

# Figure 3: other sectors post-2015
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharex=True)
for ax, (y, t) in zip(axes, [("ic_f3_s", "Insurers"), ("if_f3_s", "Investment funds"), ("bk_f3_s", "Banks")]):
    irf(ax, S[f"post_{y}"], title=t)
    ax.set_xlabel("Quarters after shock")
axes[0].set_ylabel("Net purchases of debt\nsecurities, % of assets")
fig.tight_layout()
fig.savefig(O + "fig_sectors.pdf")

# Figure 4: 2022-23 euro area aggregate by counterpart sector
eaf = pd.read_csv(O + "ea_icb_flows.csv", parse_dates=["date"]).set_index("date")
cols = {"ea_U2_S13": "Euro area\ngovernments", "ea_U2_S12K": "Euro area\nbanks", "ea_U2_S11": "Euro area\nnon-fin. corps",
        "ea_U2_S12O": "Euro area other\nfinancials", "ea_U2_S1W": "Rest of\nthe world"}
a = eaf.loc["2022-09-30":"2023-12-31", list(cols)].sum() / 1000
b = eaf.loc["2016-03-31":"2021-12-31", list(cols)].sum() / 1000 / 24 * 6
fig, ax = plt.subplots(figsize=(7, 2.8))
x = np.arange(len(cols))
ax.bar(x - 0.18, b.values, 0.36, color="0.7", label="2016–2021, average per six quarters")
ax.bar(x + 0.18, a.values, 0.36, color="0.2", label="2022Q3–2023Q4")
ax.set_xticks(x)
ax.set_xticklabels(list(cols.values()), fontsize=7.5)
ax.axhline(0, color="0.4", lw=0.6)
ax.set_ylabel("EUR billion")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(O + "fig_ea_2022.pdf")
pd.DataFrame({"tightening": a, "average": b}).round(1).to_csv(O + "ea_2022.csv")


def cell(v):
    b_, se = v
    t = abs(b_ / se)
    s = "^{***}" if t > 2.58 else "^{**}" if t > 1.96 else "^{*}" if t > 1.645 else ""
    return f"${b_:.2f}{s}$", f"$({se:.2f})$"


def rows(spec, hs=(0, 1, 2, 4, 6, 8)):
    out = []
    for name, r, key in spec:
        cs = [cell(r[str(h)][key]) for h in hs]
        out.append(f"{name} & " + " & ".join(c[0] for c in cs) + f" & {r['0']['N']} \\\\")
        out.append(" & " + " & ".join(c[1] for c in cs) + " & \\\\[2pt]")
    return "\n".join(out)


open(O + "tab_main.tex", "w").write(
    "\\multicolumn{8}{l}{\\textit{Panel A: 1999--2014}} \\\\\n" + rows([
        ("Debt securities", S["pre_ic_f3_s"], "mp"), ("\\quad long-term", S["pre_ic_f3l_s"], "mp"),
        ("Investment fund shares", S["pre_ic_f52_s"], "mp"), ("Deposits", S["pre_ic_f2_s"], "mp"),
        ("Technical reserves (liability)", S["pre_ic_f6_s"], "mp")]) +
    "\n\\midrule\n\\multicolumn{8}{l}{\\textit{Panel B: 2015--2025}} \\\\\n" + rows([
        ("Debt securities", S["post_ic_f3_s"], "mp"), ("\\quad long-term", S["post_ic_f3l_s"], "mp"),
        ("Investment fund shares", S["post_ic_f52_s"], "mp"), ("Deposits", S["post_ic_f2_s"], "mp"),
        ("Technical reserves (liability)", S["post_ic_f6_s"], "mp")]))
open(O + "tab_full.tex", "w").write(rows([
    ("Debt securities", B["base_ic_f3_s"], "mp"), ("Investment fund shares", B["base_ic_f52_s"], "mp"),
    ("Equity", B["base_ic_f5_s"], "mp"), ("Deposits", B["base_ic_f2_s"], "mp"),
    ("Technical reserves (liability)", B["base_ic_f6_s"], "mp")]))
PI = json.load(open(O + "preint.json"))
open(O + "tab_guar.tex", "w").write(
    "\\multicolumn{8}{l}{\\textit{Panel A: 1999--2014}} \\\\\n" + rows([
        ("Debt securities", PI["preint_ic_f3_s"], "mp_x"), ("Technical reserves", PI["preint_ic_f6_s"], "mp_x"),
        ("Investment fund shares", PI["preint_ic_f52_s"], "mp_x")]) +
    "\n\\midrule\n\\multicolumn{8}{l}{\\textit{Panel B: 2015--2025}} \\\\\n" + rows([
        ("Debt securities", S["postint_ic_f3_s"], "mp_x"), ("Technical reserves", S["postint_ic_f6_s"], "mp_x"),
        ("Investment fund shares", S["postint_ic_f52_s"], "mp_x")]))
open(O + "tab_sectors.tex", "w").write(
    "\\multicolumn{8}{l}{\\textit{1999--2014}} \\\\\n" + rows([
        ("Insurers", S["pre_ic_f3_s"], "mp"), ("Investment funds", S["pre_if_f3_s"], "mp"),
        ("Banks", S["pre_bk_f3_s"], "mp")]) +
    "\n\\midrule\n\\multicolumn{8}{l}{\\textit{2015--2025}} \\\\\n" + rows([
        ("Insurers", S["post_ic_f3_s"], "mp"), ("Investment funds", S["post_if_f3_s"], "mp"),
        ("Banks", S["post_bk_f3_s"], "mp")]))
open(O + "tab_robust.tex", "w").write(rows([
    ("Baseline, 2015--2025", S["post_ic_f3_s"], "mp"),
    ("Excluding 2022--2023", X["post_ex2223_ic_f3_s"], "mp"),
    ("Solvency II period, 2016--2025", X["s2_ic_f3_s"], "mp"),
    ("Tightening shocks only", X["post_tight_ic_f3_s"], "mp"),
    ("Easing shocks only", X["post_ease_ic_f3_s"], "mp"),
    ("Full sample, median-rotation shock", RB["median_ic_f3_s"], "mp"),
    ("Full sample, excluding 2020", RB["nocovid_ic_f3_s"], "mp"),
    ("Full sample, seven largest markets", RB["big_ic_f3_s"], "mp"),
    ("Non-euro EU countries, 2015--2025", X["noneu_post_ic_f3_s"], "mp")]))
open(O + "tab_break.tex", "w").write(rows([
    ("Debt securities", X["break_ic_f3_s"], "mp_x"), ("Technical reserves", X["break_ic_f6_s"], "mp_x")]))
ep = pd.read_csv(O + "episode.csv", index_col=0)
open(O + "tab_episode.tex", "w").write("\n".join(
    f"{i} & " + " & ".join(f"${v:.2f}$" for v in r[["ic_f3", "ic_f52", "ic_f5", "ic_f2", "ic_f6"]].values) + " \\\\"
    for i, r in ep.iterrows()))
expo = pd.read_csv(O + "exposure.csv", index_col=0)
expo["guar_share"] = expo["guar_share"].clip(0, 1)
open(O + "tab_expo.tex", "w").write(" \\\\\n".join(
    " & ".join([c, f"{expo.loc[c, 'guar_share']:.2f}", f"{expo.loc[c, 'life_share']:.2f}"]) for c in
    expo.sort_values("guar_share", ascending=False).index) + " \\\\")
d = P[ea]
desc = []
for c, nm in [("ic_f3_s", "Debt securities"), ("ic_f3l_s", "\\quad long-term"), ("ic_f52_s", "Investment fund shares"),
              ("ic_f5_s", "Equity"), ("ic_f2_s", "Deposits"), ("ic_f6_s", "Technical reserves (liability)"),
              ("if_f3_s", "Investment funds: debt securities"), ("bk_f3_s", "Banks: debt securities")]:
    s = d[c + "_w"].dropna()
    desc.append(f"{nm} & {len(s)} & {s.mean():.2f} & {s.std():.2f} & {s.quantile(.1):.2f} & {s.quantile(.9):.2f} \\\\")
s = P.drop_duplicates("date")
s = s[(s["date"] <= "2025-12-31")]["mp"].dropna()
desc.append(f"Monetary policy shock, 10 bp units & {len(s)} & {s.mean():.2f} & {s.std():.2f} & {s.quantile(.1):.2f} & {s.quantile(.9):.2f} \\\\")
open(O + "tab_desc.tex", "w").write("\n".join(desc))
print(open(O + "tab_main.tex").read()[:800])
print(pd.read_csv(O + "ea_2022.csv"))
