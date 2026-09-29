"""Figure 1 of the ACL paper: Jev minus each 7B rescorer, per participant and pooled."""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

A = "../analysis/"; t15 = json.load(open(A + "t15_prereg_results.json")); t12 = json.load(open(A + "t12_prereg_results.json"))
pool = json.load(open(A + "pooled_exploratory.json"))
rows = [("jev-opt/A", "vs OPT-6.7b, A"), ("jev-qwen/A", "vs Qwen2.5-7B, A"),
        ("jev-opt/B", "vs OPT-6.7b, B"), ("jev-qwen/B", "vs Qwen2.5-7B, B")]
series = [("T15", t15["all"]["primary"], "#2f6db0", "o", t15["margin_delta"]),
          ("T12 (pre-registered)", t12["all"]["primary"], "#d9822b", "s", t12["margin_delta"]),
          ("pooled (exploratory)", pool, "#222222", "D", None)]
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(3.3, 2.6))
for i, (key, lab) in enumerate(rows):
    y0 = len(rows) - 1 - i
    for j, (name, res, col, mk, delta) in enumerate(series):
        y = y0 + 0.22 - j * 0.22; r = res[key]
        ax.plot(r["ci"], [y, y], color=col, lw=1.2); ax.plot(r["diff"], y, mk, color=col, ms=3.8)
        if delta is not None: ax.plot([delta, delta], [y - 0.08, y + 0.08], color=col, lw=1.6)
ax.axvline(0, color="#888888", lw=0.8, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, lab in rows][::-1])
ax.set_xlabel("WER(Jev) $-$ WER(7B), points (95% CI)"); ax.set_xlim(-1.6, 1.0)
ax.text(-1.55, len(rows) - 0.45, "Jev better", fontsize=8, color="#555555", va="center")
ax.text(0.95, len(rows) - 0.45, "7B better", fontsize=8, color="#555555", va="center", ha="right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
handles = [Line2D([], [], color=c, marker=m, lw=1.2, ms=4, label=n) for n, _, c, m, _ in series]
handles.append(Line2D([], [], color="#999999", lw=1.6, marker="|", ms=7, ls="", label="margin $\\delta$ (per participant)"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.25), ncol=2, fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "fig_forest.pdf", bbox_inches="tight")
print("ok")
