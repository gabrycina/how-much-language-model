"""Emit the Kaldi-regime paper fragments. No number is transcribed by hand."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("real/results_kaldi.json"))
D = json.load(open("real/diagnostics_kaldi.json"))
V = json.load(open("real/revision_kaldi.json"))
AB = json.load(open("real/ablation_normalised.json"))

NAME = {"none": "No rescoring (first pass only)",
        "opt": "OPT-6.7b", "qwen": "Qwen2.5-7B", "gpt2": "GPT-2 large",
        "gpt2medium": "GPT-2 medium", "gpt2small": "GPT-2 small",
        "distilgpt2": "DistilGPT-2",
        "optchoice": "OPT-6.7b", "qwenchoice": "Qwen2.5-7B",
        "jev": "Jev (hosted)", "laya": "Laya"}
PER = ["opt", "qwen", "gpt2", "gpt2medium", "gpt2small", "distilgpt2"]
TYPED = ["jev", "laya", "optchoice", "qwenchoice"]
SHORT = {"none": "no rescoring", "opt": "OPT-6.7b", "qwen": "Qwen-7B",
         "gpt2": "GPT-2 large", "gpt2medium": "GPT-2 medium", "gpt2small": "GPT-2 small",
         "distilgpt2": "DistilGPT-2", "jev": "Jev (typed)", "laya": "Laya (typed)",
         "optchoice": "OPT-6.7b (typed)", "qwenchoice": "Qwen-7B (typed)"}
DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
       "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def key(s):
    o = "".join(DIG.get(c, c) for c in str(s) if c.isalnum())
    return o[:1].upper() + o[1:]


def main_table():
    rows = {r["arm"]: r for r in R["rows"]}
    L = [r"\begin{tabular}{llrlrrr}", r"\toprule",
         r"Rescorer & Shape & WER & 95\% CI & Sent.\ acc. & Params & TFLOP \\", r"\midrule",
         r"\multicolumn{7}{l}{\emph{Per-candidate scoring}} \\"]
    for k in PER:
        if k not in rows:
            continue
        r = rows[k]
        b = r"\bf " if k in ("opt", "gpt2small") else ""
        L.append(f"\\quad {b}{NAME[k]} & & {b}{r['wer']:.1f} & "
                 f"[{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {r['sent_acc']:.1f} & "
                 f"{r['params']/1e6:.0f}M & {r['tflops']:.2f} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{7}{l}{\emph{Typed selection (one pass over the whole list)}} \\")
    for k in TYPED:
        if k not in rows:
            continue
        r = rows[k]
        p = f"{r['params']/1e6:.0f}M" if r["params"] else "---"
        tf = f"{r['tflops']:.2f}" if r["tflops"] else "---"
        L.append(f"\\quad {NAME[k]} & & {r['wer']:.1f} & "
                 f"[{r['ci'][0]:.1f}, {r['ci'][1]:.1f}] & {r['sent_acc']:.1f} & {p} & {tf} \\\\")
    n = rows["none"]
    L += [r"\midrule",
          f"{NAME['none']} & --- & {n['wer']:.1f} & "
          f"[{n['ci'][0]:.1f}, {n['ci'][1]:.1f}] & {n['sent_acc']:.1f} & --- & --- \\\\",
          rf"\emph{{Oracle (best in list)}} & --- & \emph{{{R['oracle_wer']:.1f}}} & --- & --- & --- & --- \\",
          r"\bottomrule", r"\end{tabular}"]
    open("paper_real/tab_main.tex", "w").write("\n".join(L) + "\n")


def sig_table():
    L = [r"\begin{tabular}{lrlr}", r"\toprule",
         r"Comparison (A vs.\ B) & $\Delta$WER & 95\% CI & $p$ \\", r"\midrule"]
    for k, v in R["significance"].items():
        a, b = k.split("_vs_")
        p = "$<$0.001" if v["p"] < 0.001 else (
            "$>$0.999" if v["p"] > 0.999 else f"{v['p']:.3f}")
        L.append(f"{NAME.get(a,a)} ({'typed' if a in TYPED else 'per-cand.'}) vs.\\ "
                 f"{NAME.get(b,b)} & {v['delta_wer']:+.2f} & "
                 f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] & {p} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open("paper_real/tab_sig.tex", "w").write("\n".join(L) + "\n")


def collapse_table():
    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Typed rescorer & Labels used & Mass on one label & Truth selected \\",
         r"\midrule"]
    for a in TYPED:
        d = D[a]
        L.append(f"{NAME[a]} & {d['distinct_labels']} / 100 & "
                 f"{d['frac_most_common']*100:.0f}\\% & {d['truth_picked_pct']:.1f}\\% \\\\")
    L += [r"\midrule",
          rf"\emph{{First-pass top-1 (no rescorer)}} & --- & --- & "
          rf"\emph{{{D['first_pass_top1_pct']:.1f}\%}} \\",
          r"\bottomrule", r"\end{tabular}"]
    open("paper_real/tab_collapse.tex", "w").write("\n".join(L) + "\n")


def abstention_table():
    S = V["softmax_abstention"]
    cov = [30, 50, 70, 90, 100]
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r" & & & \multicolumn{5}{c}{sentence error among spoken, at coverage} \\",
         r"\cmidrule(lr){4-8}",
         r"Rescorer & Shape & ECE & " + " & ".join(f"{c}\\%" for c in cov) + r" \\",
         r"\midrule"]
    order = ["jev", "laya", "optchoice", "qwenchoice", "opt", "gpt2", "qwen"]
    for a in order:
        if a not in S:
            continue
        c = S[a]
        shape = "typed" if a in TYPED else "per-cand.\\ + softmax"
        b = r"\bf " if a == "jev" else ""
        L.append(f"{b}{NAME.get(a,a)} & {shape} & {c['ece']:.3f} & " + " & ".join(
            f"{b}{c['curve'][str(k)]:.0f}\\%" for k in cov) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open("paper_real/tab_abstention.tex", "w").write("\n".join(L) + "\n")


def numbers():
    rows = {r["arm"]: r for r in R["rows"]}

    def mac(n, v):
        return "\\newcommand{\\" + n + "}{" + v + "}"

    L = [mac("nTest", str(R["n_test"])), mac("nDev", str(R["n_dev"])),
         mac("oracleWer", f"{R['oracle_wer']:.1f}"),
         mac("truthInList", f"{R['truth_in_nbest']:.0f}"),
         mac("nLists", str(R["list_meta"]["n"]))]
    KEY = {"none": "None", "opt": "Opt", "qwen": "Qwen", "gpt2": "Gpt",
           "gpt2medium": "GptMed", "gpt2small": "GptSmall", "distilgpt2": "Distil",
           "jev": "Jev", "laya": "Laya", "optchoice": "OptTyped",
           "qwenchoice": "QwenTyped"}
    for k, kk in KEY.items():
        if k not in rows:
            continue
        r = rows[k]
        L += [mac(f"wer{kk}", f"{r['wer']:.1f}"), mac(f"acc{kk}", f"{r['sent_acc']:.1f}")]
        if r["tflops"]:
            L.append(mac(f"tflop{kk}", f"{r['tflops']:.2f}"))
        if r["params"]:
            L.append(mac(f"par{kk}", f"{r['params']/1e6:.0f}M"))
    for a in TYPED:
        L += [mac(f"lab{key(a)}", str(D[a]["distinct_labels"])),
              mac(f"truthPick{key(a)}", f"{D[a]['truth_picked_pct']:.1f}")]
    L.append(mac("firstPassTopOne", f"{D['first_pass_top1_pct']:.1f}"))
    WORD = {30: "Thirty", 50: "Fifty", 70: "Seventy", 90: "Ninety", 100: "Hundred"}
    for arm in ("jev", "opt", "gpt2"):
        c = V["softmax_abstention"].get(arm)
        if not c:
            continue
        for k, w in WORD.items():
            if str(k) in c["curve"]:
                L.append(mac(f"risk{key(arm)}At{w}", f"{c['curve'][str(k)]:.0f}"))
    tok_pc = V["tokens"]["per_candidate_tokens_per_decision_est"]
    L += [mac("tokPerCand", f"{tok_pc:.0f}"),
          mac("tokTypedOpt",
              f"{V['tokens'].get('optchoice_prompt_tokens_median', 0):.0f}"),
          mac("tokTypedQwen",
              f"{V['tokens'].get('qwenchoice_prompt_tokens_median', 0):.0f}")]
    ab = AB["conditional"]
    L += [mac("qwenTypedAtFive", f"{ab['qwenchoice']['5']['conditional']:.1f}"),
          mac("qwenTypedAtSixty", f"{ab['qwenchoice']['60']['conditional']:.1f}"),
          mac("optTypedAtSixty", f"{ab['optchoice']['60']['conditional']:.1f}"),
          mac("baseAtFive",
              f"{AB['first_pass_top1_pct']/AB['availability']['5']*100:.1f}"),
          mac("baseAtSixty",
              f"{AB['first_pass_top1_pct']/AB['availability']['60']*100:.1f}"),
          mac("qwenBorda", f"{AB['borda']['qwenchoice']['borda']:.1f}"),
          mac("optBorda", f"{AB['borda']['optchoice']['borda']:.1f}")]
    open("paper_real/numbers.tex", "w").write("\n".join(L) + "\n")


OFFSET = {"distilgpt2": (7, 9), "gpt2small": (2, -14), "gpt2medium": (4, 9),
          "gpt2": (8, -3), "opt": (8, -3), "qwen": (8, 1),
          "jev": (-7, 6), "laya": (5, -13), "optchoice": (7, 4),
          "qwenchoice": (7, -12)}
PEN, TRACE, WARN, GOOD, GREY = "#C63F26", "#2E6E8E", "#9A6C09", "#1E6F52", "#8A96A0"


def fig_accuracy_vs_cost():
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "axes.edgecolor": "#444", "axes.linewidth": 0.7,
                         "figure.dpi": 200, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02})
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for r in R["rows"]:
        if r["tflops"] is None or r["arm"] == "none":
            continue
        col = TRACE if r["shape"] == "per-candidate" else PEN
        mk = "o" if r["shape"] == "per-candidate" else "^"
        ax.errorbar(r["tflops"], r["wer"],
                    yerr=[[r["wer"] - r["ci"][0]], [r["ci"][1] - r["wer"]]],
                    fmt=mk, ms=5, color=col, elinewidth=0.8, capsize=2)
        ax.annotate(SHORT.get(r["arm"], r["arm"]), (r["tflops"], r["wer"]),
                    textcoords="offset points",
                    xytext=OFFSET.get(r["arm"], (6, -3)),
                    ha=("center" if False else
                        "right" if OFFSET.get(r["arm"], (6, 0))[0] < 0 else "left"),
                    fontsize=6.3, color=col)
    base = [r for r in R["rows"] if r["arm"] == "none"][0]
    ax.axhline(base["wer"], color=GREY, ls="--", lw=1)
    ax.text(0.02, base["wer"] + 0.06, "no rescoring", fontsize=6.5, color=GREY)
    ax.axhline(R["oracle_wer"], color=GOOD, ls=":", lw=1)
    ax.text(0.02, R["oracle_wer"] + 0.06, "oracle", fontsize=6.5, color=GOOD)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 8)
    ax.set_xlabel("TFLOPs per decision (log)")
    ax.set_ylabel("word error rate (\%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("paper_real/fig_cost.pdf")
    plt.close(fig)


def fig_label_collapse():
    arms = ["jev", "laya", "optchoice", "qwenchoice"]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    xs = np.arange(len(arms))
    dist = [D[a]["distinct_labels"] for a in arms]
    frac = [D[a]["frac_most_common"] * 100 for a in arms]
    ax.bar(xs - 0.2, dist, 0.4, color=TRACE, label="distinct labels used (of 100)")
    ax.bar(xs + 0.2, frac, 0.4, color=WARN, label="\% mass on single label")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[a].replace(" (typed)", "") for a in arms], fontsize=7)
    ax.legend(fontsize=6.2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("paper_real/fig_collapse.pdf")
    plt.close(fig)


def fig_abstention():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for arm, c in R["calibration"].items():
        col = {"jev": PEN, "laya": TRACE, "optchoice": WARN,
               "qwenchoice": GREY}.get(arm, GREY)
        ax.plot([p["coverage"] * 100 for p in c["curve"]],
                [p["risk"] * 100 for p in c["curve"]],
                marker="o", ms=3, lw=1.3, color=col, label=SHORT.get(arm, arm))
    ax.set_xlabel("coverage: \% of sentences spoken")
    ax.set_ylabel("sentence error among spoken (\%)")
    ax.legend(fontsize=6.2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("paper_real/fig_abstention.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main_table()
    sig_table()
    collapse_table()
    abstention_table()
    numbers()
    fig_accuracy_vs_cost()
    fig_label_collapse()
    fig_abstention()
    print("wrote Kaldi paper_real fragments + figures")
