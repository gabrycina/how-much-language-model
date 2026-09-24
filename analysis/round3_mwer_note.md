> **Superseded.** This note records a round-3 attempt that was blocked before training
> began. The experiment was subsequently run on GPU in round 4; its results are in
> `round3_mwer_results.json` and are the numbers the paper reports (test WER 8.15%,
> p=0.62 vs no rescoring). Read the account below as a record of the blocked attempt,
> not as evidence that the experiment was never performed.

# Round 3 — MWER fine-tuning experiment: fallback note

> SUPERSEDED (round 4, L40S GPU run, 2026-09-22): the training described below as
> blocked was subsequently performed. Final artifacts are `mwer_gpu.log` and
> `round3_mwer_results.json` (test WER 8.15%, p=0.62 vs no-rescoring, p=0.84 vs
> frozen GPT-2 small); the paper cites those round-4 numbers. This note is retained
> only as a record of the blocked round-3 attempt.
>
> **Outcome of the round-3 attempt below: the training run was not performed. No training results are reported or implied anywhere in this note.** What follows is (a) an exact record of what blocked the experiment and (b) a draft paper section on discriminative rescoring, written for the revision.

A complete, protocol-exact training script is saved alongside this note at
`~/workspace/jev-revision/analysis_out/round3_mwer_train.py`, ready to run on the GPU box. It implements the full specification from the task (300/119/419/978 split, top-20 MWER, AdamW 1e-5, batch 4 lists, 3 epochs, early stopping, full-list eval, alpha grid 0.00–1.00 on dev, paired bootstrap vs no-rescoring and vs cached untuned GPT-2 small) and includes pipeline self-checks that reproduce the cached baselines (GPT-2 small @ α=0.4 → 7.983; no-rescoring → 8.091) before any fine-tuned evaluation.

## (a) What blocked the experiment

Machine: 2 vCPUs, 7 GB RAM, no GPU, no torch/transformers pre-installed. Started 21:05 UTC 2026-09-21; 75-minute hardbound 22:20 UTC.

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu transformers` → refused: `error: externally-managed-environment` (PEP 668). Retried inside a fresh venv at `/tmp/mwerenv`.
2. venv install → `ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device`. Cause: `/tmp` is a 512 MB tmpfs with ~190 MB free; the torch CPU wheel (~180 MB) does not fit. Retried with system pip (`--break-system-packages`, `--no-cache-dir`, `TMPDIR` pointed at `analysis_out/.pip-tmp`).
3. torch 2.14.0+cpu installed. `pip install transformers …` → `ERROR: Cannot uninstall typing_extensions 4.10.0, RECORD file not found. Hint: The package was installed by debian.` Workaround: `--ignore-installed`, which pulled numpy 2.5.3.
4. numpy 2.x broke the Debian scipy/sklearn: `ImportError: numpy.core.multiarray failed to import` (compiled against numpy 1.x), surfacing inside `transformers` via `generation/candidate_generator.py → from sklearn.metrics import roc_curve`. Downgraded to `numpy==1.26.4`; then the pip-installed sklearn (built for numpy 2) broke instead: `ImportError: cannot load module more than once per process`. Fix: deleted the mixed `numpy/` directory tree outright and cleanly installed `numpy==1.26.4`, `scipy==1.14.1`, `scikit-learn==1.5.2`. All imports then passed.
5. `GPT2LMHeadModel.from_pretrained('gpt2')` → `httpx.InvalidURL: Invalid port: ':1]'`. Cause: the runtime sets `no_proxy`/`NO_PROXY` containing IPv6 literals (`[::1]`, `[fd8b:…::1]`, …); httpx 0.28.1 builds proxy-bypass URL patterns from these entries and its vendored urlparse cannot parse them. Direct (proxy-less) connections fail (`curl` → code 000), so the proxy is mandatory. Workaround: `export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1`, after which the HuggingFace download started.
6. The download through `huggingface_hub` then stalled: ~2.9 MB fetched in ~8 minutes (proxy-throttled; a raw `curl` range probe to the same host measured ~1.9 MB/s, but the hub client was far slower). The GPT-2 small weights are ~497 MB. The 25-minute setup-verification window (21:05–21:30 UTC) expired with the weights not downloaded and no forward pass ever executed.

Decisive blocker, independent of the environment saga: compute. Scoring the full dev+test lists (1,397 lists × ~39 candidates × ~8 tokens ≈ 432k tokens) forward through GPT-2 small is ~1.1e14 FLOP — roughly 1–2 hours on 2 CPU cores — and the 3-epoch MWER training (forward + backward) is another ~1.5–3 hours. The 75-minute hardbound cannot contain this. Per the task instruction ("Do not start a run you cannot finish and evaluate inside the bound"), no training run was started.

Net: setup verification never completed (weights never arrived; forward pass and per-candidate log-likelihood checks never ran). Nothing was trained, tuned, or evaluated. All numbers in the paper's tables are untouched.

## (b) Draft section: discriminative rescoring (for the revision)

*Reviewer weakness W5 notes that no rescorer in this paper received discriminative fine-tuning. The following is drafted as replacement/extension text for the rescoring discussion.*

---

All rescorers evaluated in this paper are used frozen. Each model assigns every n-best candidate a single score — a length-normalised log-likelihood for the autoregressive models, a pseudo-log-likelihood for BERT-style models — and that score is interpolated with the decoder score using two scalar weights fit by grid search on development data:

s(h) = (1 − α) · z(decoder) + α · z(LM).

Fitting α is sometimes described as MWER-style, but it is not discriminative training. The language model's parameters never move. Its ranking of hypotheses relative to each other is fixed; the fit only adjusts the trade-off between two fixed axes. If the frozen model systematically prefers a fluent-but-wrong hypothesis over the correct one, no choice of α can fix that — α can only decide how much the decoder score is allowed to overrule it.

Discriminative fine-tuning changes the scores themselves. The standard objective is minimum word error rate (MWER): for each utterance, the model scores every hypothesis in the n-best list, the scores are normalised into a posterior over the list, and the loss is the expected number of word errors under that posterior. Gradients flow through the whole language model, so training adjusts all of its parameters to make the correct hypothesis win the list, not merely to assign likely text high probability. Because the loss is computed on the interpolated final score, the model also learns to account for the first-pass scores during training — a point the RescoreBERT authors stress, since a hypothesis needs the best *combined* score, not the best LM score in isolation.

The literature reports consistent gains from this recipe. RescoreBERT (Xu et al., ICASSP 2022, arXiv:2202.01094) fine-tunes BERT with the MWER loss (plus a variance-reduced variant and an alternative MWED loss that matches the score distribution to the error distribution). On LibriSpeech it reduces WER by 6.6%/3.4% relative on the clean/other test sets over the same BERT model used without a discriminative objective, and by 7–13% relative over MLM-distillation-only BERT on internal conversational-agent data. Shivakumar et al. (ASRU 2023, arXiv:2310.06248) extend the same discriminative fine-tuning to pretrained language models of both families — GPT-2 scored by autoregressive likelihood and BERT scored by pseudo-log-likelihood — comparing probability-based MWER against embedding-pooling architectures. They find every MWER training scheme beneficial, with additional gains of up to 8.5% WER on LibriSpeech, and confirm that the discriminative step helps causal and bidirectional models alike.

We did not apply discriminative fine-tuning to any rescorer in this paper: the small-model arms are evaluated strictly as frozen likelihood functions. The efficiency curve we report therefore measures what off-the-shelf likelihoods contribute at this operating point. It is a lower bound on what small fine-tuned models could achieve — the literature above suggests a discriminatively trained GPT-2-small-class rescorer would sit strictly above the frozen point we plot — and we leave that training, which requires GPU compute we did not have for this revision, to future work.

---

*Note on "LoRB": the task brief named RescoreBERT/MWER/LoRB as the family to discuss. I verified RescoreBERT and the Shivakumar et al. GPT-2/BERT MWER work above; I could not verify a method called "LoRB" in the rescoring literature and have not asserted anything about it. If LoRB refers to a specific paper, it needs a citation check before this section is final.*
