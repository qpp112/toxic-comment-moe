# Toxic comment classification: do a Mixture-of-Experts head and class-balanced losses help?

[![tests](https://github.com/qpp112/toxic-comment-moe/actions/workflows/tests.yml/badge.svg)](https://github.com/qpp112/toxic-comment-moe/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qpp112/toxic-comment-moe/blob/main/notebooks/train_on_colab.ipynb)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Multi-label toxicity detection on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
(Wikipedia talk-page comments, 6 overlapping labels; `threat` is 0.3% of comments). In a course
project (UC Irvine CS178, Fall 2025), reweighted losses seemed to help BERT with the rare labels
and a Mixture-of-Experts (MoE) head looked promising. This repository rebuilds that project
properly and tests both ideas in a controlled study: **2 classification heads × 4 losses × 3 seeds
= 24 BERT-base runs** on the full training set (160k comments), scored on Kaggle's official
labelled test set (64k comments), with decision thresholds tuned on a validation split only.
A follow-up of **27 more runs** works out why the MoE head failed, tests two heads built to get
around it, and tests longer inputs.

**Result: no head and no loss beats plain BERT + BCE, including the two heads built to fix the
MoE's problem. Letting BERT read more of each comment does.**

* **All 8 settings tie on ranking quality.** Test ROC-AUC spans 0.9850–0.9858 (baseline
  0.9855 ± 0.0001). Plain BCE has the best PR-AUC; every alternative is 0.005–0.024 lower. On
  macro-F1 and rare-label F1, no gain is distinguishable from seed noise: the largest, +0.017 ±
  0.015 rare-label F1 for MoE + weighted BCE, is about one standard deviation.
* **Reweighting mostly moves the decision threshold.** Weighted BCE pushes the best thresholds to
  0.89–0.998 (plain BCE: 0.18–0.70). At a fixed 0.5 threshold it is far worse than BCE (macro-F1
  0.50 vs 0.62); with thresholds tuned on validation the two tie (0.622 vs 0.617). The course
  project's gain, measured at 0.5 on 1/20 of the data (where plain BCE scored F1 = 0 on the three
  rarest labels), does not survive the full data and tuned thresholds.
* **The MoE router learns toxic vs. clean, not kinds of toxicity.** In all 12 MoE runs, comments
  with any toxic label send on average 90% of their routing weight to the same two of six experts,
  whatever the label. It cannot do better: 93–100% of comments with a rare label are also `toxic`,
  so one routing decision per comment cannot tell the labels apart.
* **Fixing the routing does not fix the scores.** A multi-gate MoE (one gate per label) does learn
  label-specific experts, and label-wise attention gives every label its own view of the words.
  Both tie with the linear head (ROC-AUC +0.0002 and −0.0001, paired over 3 seeds).
* **Once BERT is fine-tuned, the head hardly matters.** On a frozen encoder, label-wise attention
  beats a linear head by 0.026 ROC-AUC. Fine-tuned, all five heads are within 0.0002 of the
  linear head. The MoE head's small loss on rare-label PR-AUC (−0.014) is matched by an MLP of the same
  size (−0.017), so it comes from the extra layer, not the routing.
* **What helps is reading more of each comment.** 19% of test comments are longer than the
  128-token input. Raising it to 512 tokens adds +0.0014 ± 0.0002 ROC-AUC and +0.006 ± 0.002
  PR-AUC (better on every seed) for under 10% more training time, reaching 0.9868. Keeping the
  first 63 and last 63 tokens instead of the first 126 gives +0.0011 ± 0.0002 at no extra cost.
  No head or loss raised ROC-AUC by more than 0.0003, and averaging 3 seeds by 0.0004. One
  caveat: these runs used a newer `transformers` release (5.18 vs 5.17), so a same-environment
  re-run of the baseline is queued as a control.
* **Where the rest of the headroom is.** Averaging 3 seeds adds +0.008 PR-AUC. The models also
  agree on many of their mistakes: averaged over 36 runs, 730 test comments labelled clean get
  p(`toxic`) ≥ 0.9, which looks like label noise. [`docs/diagnosis.md`](docs/diagnosis.md) has
  the full analysis.

[`docs/course_project.md`](docs/course_project.md) describes the original project and the bugs
this version fixes.

## Results

<!-- RESULTS:START -->
Trained on 143,613 comments (100% of `train.csv` minus a 15,958-comment validation split) and evaluated on the **63,978-comment official Kaggle test set**. F1 thresholds are tuned per label on the validation split, never on test. Every setting is trained with 3 seeds (42, 43, 44) on the same data split; cells show mean ± sample std over seeds. Best mean per column in bold; the paired table shows which differences are larger than seed-to-seed noise.

**Difference from the baseline (BERT + linear head + BCE), paired by seed.** Positive = better than the baseline. Mean ± std over 3 seeds.

| Head | Loss | Δ ROC-AUC | Δ PR-AUC | Δ Rare-label PR-AUC | Δ Macro-F1 (val-tuned thr.) | Δ Rare-label F1 |
|---|---|---:|---:|---:|---:|---:|
| Linear head | Weighted BCE | -0.0004 ± 0.0003 | -0.022 ± 0.005 | -0.023 ± 0.008 | +0.005 ± 0.012 | +0.013 ± 0.022 |
| Linear head | Focal | +0.0003 ± 0.0001 | -0.005 ± 0.002 | -0.009 ± 0.006 | -0.003 ± 0.016 | -0.003 ± 0.023 |
| Linear head | Class-balanced focal | +0.0001 ± 0.0002 | -0.012 ± 0.002 | -0.020 ± 0.005 | -0.001 ± 0.018 | +0.001 ± 0.026 |
| MoE head | BCE | +0.0000 ± 0.0001 | -0.008 ± 0.001 | -0.014 ± 0.002 | +0.002 ± 0.013 | +0.001 ± 0.014 |
| MoE head | Weighted BCE | -0.0002 ± 0.0001 | -0.024 ± 0.004 | -0.035 ± 0.005 | +0.007 ± 0.012 | +0.017 ± 0.015 |
| MoE head | Focal | +0.0002 ± 0.0000 | -0.009 ± 0.002 | -0.016 ± 0.004 | +0.001 ± 0.011 | +0.000 ± 0.013 |
| MoE head | Class-balanced focal | +0.0000 ± 0.0002 | -0.010 ± 0.006 | -0.018 ± 0.012 | +0.003 ± 0.013 | +0.007 ± 0.012 |

**Routing.** Across all 12 MoE runs, comments carrying any toxic label put on average 90% (minimum 61%) of their router weight on the same two experts, whatever the kind of toxicity, while clean comments put 29% on those two experts (uniform routing would be 33%). The router learned a toxic-vs-clean split, not experts for specific kinds of toxicity.

**Loss × head ablation (BERT-base encoder)**

| Head | Loss | Test ROC-AUC | Test PR-AUC | Macro-F1 @0.5 | Macro-F1 (val-tuned thr.) | Rare-label F1¹ |
|---|---|---:|---:|---:|---:|---:|
| Linear head | BCE | 0.9855 ± 0.0001 | **0.688 ± 0.004** | 0.618 ± 0.001 | 0.617 ± 0.009 | 0.533 ± 0.012 |
| Linear head | Weighted BCE | 0.9850 ± 0.0003 | 0.665 ± 0.008 | 0.502 ± 0.003 | 0.622 ± 0.003 | 0.546 ± 0.010 |
| Linear head | Focal | **0.9858 ± 0.0001** | 0.683 ± 0.006 | 0.618 ± 0.002 | 0.613 ± 0.007 | 0.530 ± 0.012 |
| Linear head | Class-balanced focal | 0.9856 ± 0.0001 | 0.675 ± 0.006 | 0.543 ± 0.005 | 0.615 ± 0.010 | 0.534 ± 0.016 |
| MoE head | BCE | 0.9855 ± 0.0002 | 0.680 ± 0.003 | 0.619 ± 0.002 | 0.619 ± 0.005 | 0.534 ± 0.002 |
| MoE head | Weighted BCE | 0.9853 ± 0.0002 | 0.664 ± 0.005 | 0.507 ± 0.004 | **0.624 ± 0.004** | **0.550 ± 0.006** |
| MoE head | Focal | 0.9857 ± 0.0001 | 0.679 ± 0.004 | **0.619 ± 0.002** | 0.618 ± 0.005 | 0.533 ± 0.004 |
| MoE head | Class-balanced focal | 0.9855 ± 0.0001 | 0.677 ± 0.004 | 0.548 ± 0.001 | 0.620 ± 0.004 | 0.540 ± 0.007 |

¹ Mean over the three rarest labels: `severe_toxic`, `threat`, `identity_hate`.

**Follow-up: heads built around the MoE head's failure** (BERT-base, plain BCE, each minus the linear-head baseline on the same seed; mean ± std over 3 seeds). Why these heads: `docs/diagnosis.md`.

| Head | Head params | Δ ROC-AUC | Δ PR-AUC | Δ Rare-label PR-AUC | Δ Macro-F1 (val-tuned thr.) | Δ Rare-label F1 |
|---|---|---:|---:|---:|---:|---:|
| MoE head, BCE | 1.20M | +0.0000 ± 0.0001 | -0.008 ± 0.001 | -0.014 ± 0.002 | +0.002 ± 0.013 | +0.001 ± 0.014 |
| MLP head, same size as the MoE head | 1.19M | +0.0001 ± 0.0002 | -0.008 ± 0.003 | -0.017 ± 0.006 | +0.002 ± 0.007 | +0.001 ± 0.010 |
| MoE head, no load balancing | 1.20M | +0.0000 ± 0.0001 | -0.007 ± 0.005 | -0.013 ± 0.009 | +0.005 ± 0.012 | +0.011 ± 0.013 |
| Multi-gate MoE head (one gate per label) | 1.21M | +0.0002 ± 0.0001 | -0.003 ± 0.002 | -0.006 ± 0.004 | +0.003 ± 0.011 | +0.005 ± 0.021 |
| Label-wise attention head | 0.20M | -0.0001 ± 0.0002 | +0.001 ± 0.001 | +0.002 ± 0.001 | +0.003 ± 0.016 | +0.004 ± 0.023 |

**Do the labels use different experts?** Mean distance between the expert mixtures of the six labels' positive comments (total variation; 0 = the same experts, 1 = disjoint experts): MoE router 0.13, multi-gate MoE 0.47. For one and the same comment, the multi-gate head's six label gates differ by 0.87 on average; a per-comment router scores 0 by construction.

**Frozen encoder vs fine-tuned encoder.** The same heads trained on fixed BERT-base features (`configs/probes.yaml`). A head that helps only when the encoder cannot adapt is doing work that fine-tuning already does.

| Head | Frozen: test ROC-AUC | Frozen: test PR-AUC | Frozen: Δ ROC-AUC vs linear | Fine-tuned: Δ ROC-AUC vs linear |
|---|---:|---:|---:|---:|
| Linear head | 0.9545 | 0.492 | baseline | baseline |
| Linear head, mean pooling | 0.9612 | 0.525 | +0.0067 | – |
| MLP head | 0.9614 | 0.510 | +0.0069 | +0.0001 ± 0.0002 |
| MoE head | 0.9591 | 0.506 | +0.0046 | +0.0000 ± 0.0001 |
| Multi-gate MoE head | 0.9600 | 0.518 | +0.0055 | +0.0002 ± 0.0001 |
| Label-wise attention head | 0.9803 | 0.638 | +0.0258 | -0.0001 ± 0.0002 |

**Longer inputs** (`configs/length.yaml`; linear head + BCE, each minus the 128-token baseline on the same seed; mean ± std over 3 seeds). The baseline sees the first 126 tokens of each comment and takes 7.5 min per run.

| Input | Time / run | Test ROC-AUC | Δ ROC-AUC | Δ PR-AUC | Δ Rare-label PR-AUC | Δ Macro-F1 (val-tuned thr.) | Δ Rare-label F1 |
|---|---|---|---:|---:|---:|---:|---:|
| 256 tokens | 7.7 min | 0.9866 ± 0.0001 | +0.0011 ± 0.0002 | +0.005 ± 0.003 | +0.006 ± 0.005 | +0.004 ± 0.007 | +0.012 ± 0.010 |
| 512 tokens | 8.2 min | 0.9868 ± 0.0000 | +0.0014 ± 0.0002 | +0.006 ± 0.002 | +0.006 ± 0.004 | +0.007 ± 0.009 | +0.010 ± 0.016 |
| 128 tokens: first 63 + last 63 | 7.5 min | 0.9866 ± 0.0001 | +0.0011 ± 0.0002 | +0.003 ± 0.001 | +0.000 ± 0.001 | +0.000 ± 0.015 | +0.001 ± 0.018 |

Δ test ROC-AUC per seed (42, 43, 44): 256 tokens +0.0011, +0.0013, +0.0009; 512 tokens +0.0012, +0.0016, +0.0013; 128 tokens: first 63 + last 63 +0.0010, +0.0013, +0.0010. Δ validation ROC-AUC (mean over labels): 256 tokens +0.0009 ± 0.0001; 512 tokens +0.0015 ± 0.0001; 128 tokens: first 63 + last 63 +0.0013 ± 0.0001.

Δ test ROC-AUC per label:

| Input | `toxic` | `severe_toxic` | `obscene` | `threat` | `insult` | `identity_hate` |
|---|---:|---:|---:|---:|---:|---:|
| 256 tokens | +0.0012 ± 0.0003 | +0.0012 ± 0.0012 | +0.0011 ± 0.0002 | +0.0002 ± 0.0002 | +0.0018 ± 0.0001 | +0.0011 ± 0.0006 |
| 512 tokens | +0.0015 ± 0.0002 | +0.0015 ± 0.0009 | +0.0015 ± 0.0002 | +0.0004 ± 0.0001 | +0.0021 ± 0.0002 | +0.0012 ± 0.0002 |
| 128 tokens: first 63 + last 63 | +0.0018 ± 0.0001 | +0.0016 ± 0.0010 | +0.0015 ± 0.0001 | +0.0003 ± 0.0001 | +0.0014 ± 0.0002 | +0.0000 ± 0.0001 |

Δ test PR-AUC per label:

| Input | `toxic` | `severe_toxic` | `obscene` | `threat` | `insult` | `identity_hate` |
|---|---:|---:|---:|---:|---:|---:|
| 256 tokens | +0.0039 ± 0.0021 | +0.0039 ± 0.0074 | +0.0035 ± 0.0024 | +0.0073 ± 0.0113 | +0.0070 ± 0.0016 | +0.0073 ± 0.0018 |
| 512 tokens | +0.0049 ± 0.0005 | -0.0006 ± 0.0082 | +0.0050 ± 0.0022 | +0.0031 ± 0.0046 | +0.0085 ± 0.0030 | +0.0141 ± 0.0078 |
| 128 tokens: first 63 + last 63 | +0.0070 ± 0.0004 | -0.0044 ± 0.0045 | +0.0056 ± 0.0011 | -0.0050 ± 0.0109 | +0.0068 ± 0.0010 | +0.0083 ± 0.0037 |

**Per label (mean over seeds)**

| Label | Test positives | bert_bce PR-AUC | moe_cbfocal PR-AUC | bert_bce F1 | moe_cbfocal F1 |
|---|---:|---:|---:|---:|---:|
| `toxic` | 6,090 | 0.814 ± 0.001 | 0.814 ± 0.003 | 0.692 ± 0.011 | 0.686 ± 0.020 |
| `severe_toxic` | 367 | 0.394 ± 0.004 | 0.354 ± 0.020 | 0.420 ± 0.003 | 0.401 ± 0.012 |
| `obscene` | 3,691 | 0.807 ± 0.002 | 0.801 ± 0.002 | 0.704 ± 0.005 | 0.709 ± 0.015 |
| `threat` | 211 | 0.623 ± 0.023 | 0.613 ± 0.002 | 0.572 ± 0.015 | 0.593 ± 0.012 |
| `insult` | 3,427 | 0.795 ± 0.002 | 0.794 ± 0.001 | 0.705 ± 0.011 | 0.705 ± 0.001 |
| `identity_hate` | 712 | 0.693 ± 0.001 | 0.689 ± 0.012 | 0.606 ± 0.027 | 0.626 ± 0.014 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ablation_dark.png">
  <img alt="Each setting minus the BERT + BCE baseline, paired by seed" src="docs/figures/ablation_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/heads_dark.png">
  <img alt="Follow-up heads minus the BERT + BCE baseline, paired by seed" src="docs/figures/heads_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/length_dark.png">
  <img alt="Longer inputs minus the 128-token baseline, paired by seed" src="docs/figures/length_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/probes_dark.png">
  <img alt="Head gains with a frozen vs a fine-tuned encoder" src="docs/figures/probes_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/mmoe_gates_dark.png">
  <img alt="Which experts each label's gate uses in the multi-gate MoE head" src="docs/figures/mmoe_gates_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/per_label_f1_dark.png">
  <img alt="Per-label F1, baseline vs MoE + CB-focal" src="docs/figures/per_label_f1_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/pr_curves_dark.png">
  <img alt="Precision–recall curves on the test set" src="docs/figures/pr_curves_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/expert_routing_dark.png">
  <img alt="Which experts the router uses for each label" src="docs/figures/expert_routing_light.png">
</picture>

<details><summary>Run details</summary>

| Setting | Seeds | Best epoch per seed | Min / run | Precision | GPU | `transformers` |
|---|---|---|---:|---|---|---|
| `bert_bce` | 42, 43, 44 | 2, 2, 2 | 7.5 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `bert_wbce` | 42, 43, 44 | 2, 2, 2 | 7.3 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `bert_focal` | 42, 43, 44 | 2, 2, 2 | 7.4 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `bert_cbfocal` | 42, 43, 44 | 2, 2, 2 | 7.4 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `moe_bce` | 42, 43, 44 | 2, 2, 2 | 8.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `moe_wbce` | 42, 43, 44 | 2, 2, 2 | 8.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `moe_focal` | 42, 43, 44 | 2, 2, 2 | 8.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `moe_cbfocal` | 42, 43, 44 | 2, 2, 2 | 8.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `mlp_bce` | 42, 43, 44 | 2, 2, 2 | 7.1 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `moe_bce_noaux` | 42, 43, 44 | 2, 2, 2 | 7.8 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `mmoe_bce` | 42, 43, 44 | 2, 2, 2 | 7.5 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `labelattn_bce` | 42, 43, 44 | 2, 2, 2 | 7.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `probe_linear` | 42 | 3 | 4.1 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `probe_linear_mean` | 42 | 3 | 4.1 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `probe_mlp` | 42 | 3 | 4.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `probe_moe` | 42 | 3 | 5.3 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `probe_mmoe` | 42 | 3 | 4.8 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `probe_labelattn` | 42 | 3 | 4.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.17.0 |
| `bert_bce_len256` | 42, 43, 44 | 2, 2, 2 | 7.7 | bf16 | NVIDIA A100-SXM4-80GB | 5.18.0 |
| `bert_bce_len512` | 42, 43, 44 | 2, 2, 2 | 8.2 | bf16 | NVIDIA A100-SXM4-80GB | 5.18.0 |
| `bert_bce_headtail` | 42, 43, 44 | 2, 2, 2 | 7.5 | bf16 | NVIDIA A100-SXM4-80GB | 5.18.0 |

</details>
<!-- RESULTS:END -->

## Why it didn't help

The full analysis is in [`docs/diagnosis.md`](docs/diagnosis.md); every number there comes from
[`scripts/diagnose.py`](scripts/diagnose.py) (saved test predictions and labels) or the report above.

* **Tuned thresholds absorb what reweighting does.** Up-weighting positives shifts every score
  upwards, which mainly changes where the 0.5 cut falls. Once each label gets its own threshold
  from validation data, there is little left for the loss to fix, and ranking quality (PR-AUC)
  gets slightly worse, most of all with inverse-frequency weights.
* **The router split comments, not labels, and nothing is misrouted.** 93–100% of comments with a
  rare label are also `toxic`, so one routing decision per comment can only separate toxic from
  clean. The router does that well: none of the ~6,000 comments per seed that the linear head
  scores ≥ 0.9 for `toxic` is sent away from the two "toxic" experts. But the linear head already
  makes that split.
* **Almost the same function where it counts.** At the top of each label's ranking (twice as many
  comments as the label has positives), the MoE and linear heads overlap as much as two linear-head
  seeds do (0.88–0.95). Most of their disagreement is in how they order clean comments, through
  expert-specific score offsets that no metric sees. The exception is a small loss on rare-label
  PR-AUC, which a same-size MLP shares.
* **The head is not the bottleneck.** Giving each label its own gate makes the labels use different
  experts (their gates on the same comment differ by 0.87 on a 0–1 scale) and changes nothing.
  The frozen-encoder probes show why: heads that differ a lot on fixed BERT features all tie once
  BERT is fine-tuned with them.
* **Most F1 differences are threshold noise.** With thresholds tuned on test itself (not a fair
  score, but noise-free), the seed-to-seed spread of the macro-F1 differences shrinks about 4×, and
  no MoE head beats the baseline.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/diagnosis_agreement_dark.png">
  <img alt="The MoE head agrees with the linear head at the top of every label's ranking and disagrees mostly among clean comments" src="docs/figures/diagnosis_agreement_light.png">
</picture>

The follow-up heads ([`configs/heads.yaml`](configs/heads.yaml), results in the tables above):

| Run | Head | Question | Answer |
|---|---|---|---|
| `mmoe_bce` | **Multi-gate MoE**: shared experts, one softmax gate per label ([Ma et al., 2018](https://dl.acm.org/doi/10.1145/3219819.3220007)) | Can labels pick different experts for the same comment? | Yes, and it does not help. |
| `labelattn_bce` | **Label-wise attention**: each label pools the token vectors with its own attention ([Vu et al., 2020](https://arxiv.org/abs/2007.06351)); shows which words drove each label | Is the single [CLS] vector the bottleneck? | Not after fine-tuning. It is the best head on frozen features (+0.026 ROC-AUC). |
| `mlp_bce` | One MLP with the MoE head's parameter count, no routing | Is it the capacity or the routing? | Neither helps; the MLP scores like the MoE. |
| `moe_bce_noaux` | MoE head without the load-balancing loss | Does balancing force a bad split? | No. Without it, one expert takes 96–100% of comments and nothing changes. |

## What did help

The diagnosis pointed at the input rather than the head: the baseline's `toxic` ROC-AUC is 0.957
on the 19% of test comments cut at 128 tokens, vs 0.974 on the rest.
[`configs/length.yaml`](configs/length.yaml) keeps the linear head and BCE and changes only what
BERT reads (results in the tables above). 512 tokens helps every label, most of all `insult`
(+0.0021 ROC-AUC). 256 tokens gets about 80% of that gain. The first 63 + last 63 tokens match
the first 254 on ROC-AUC, so the end of a long comment says more than its middle. Section 7 of
[`docs/diagnosis.md`](docs/diagnosis.md#7-longer-inputs-the-first-change-that-helps) compares
this with expectations written before the runs.

**Next steps:** the same-environment control for the longer-input runs and their breakdown into
short and long comments (both queued); an audit of the 730 test comments labelled clean that the
models confidently score as toxic; ensembles of 512-token models; a stronger encoder
([`configs/encoders.yaml`](configs/encoders.yaml): DeBERTa-v3-base, set up but not yet run); and
MoE layers inside the encoder, routed per token.

## The problem

Each comment can carry any subset of six labels, and the labels are highly imbalanced
(rates on `train.csv`): `toxic` ≈ 9.6%, `obscene` ≈ 5.3%, `insult` ≈ 4.9%, `severe_toxic` ≈ 1.0%,
`identity_hate` ≈ 0.9%, `threat` ≈ 0.3%. About 90% of comments are clean. That makes two common
habits misleading:

* **Accuracy is meaningless here.** About 96% of all label entries are 0, so a model that
  predicts "clean" everywhere already gets ~96% element-wise accuracy. This repo reports
  ROC-AUC (Kaggle's metric), PR-AUC and F1 instead.
* **The rare labels look hopeless at a fixed 0.5 threshold.** In the course project (1/20 of
  the data), BERT + BCE scored F1 = 0 on `severe_toxic`, `threat` and `identity_hate` at 0.5.
  That motivated the reweighted losses and the MoE head tested here.

## Approach

```mermaid
flowchart LR
    T["Comment text"] --> B["BERT-base encoder<br/>(fine-tuned, shared)"]
    B --> H["[CLS] vector h"]
    H --> R["Router<br/>g = top-2(softmax(W·h))"]
    H --> E1["Expert 1<br/>MLP → 6 logits"]
    H --> E2["Expert 2<br/>MLP → 6 logits"]
    H --> E6["Expert 6<br/>MLP → 6 logits"]
    R -- "gate weights g₁…g₆" --> S(("Σ gᵢ · Expertᵢ(h)"))
    E1 --> S
    E2 --> S
    E6 --> S
    S --> O["6 sigmoid outputs:<br/>toxic … identity_hate"]
```

**Mixture-of-Experts head** ([`models.py`](src/toxic_moe/models.py)). Six expert MLPs each
predict *all six labels*. A learned router sends each comment to its top-2 experts and mixes
their logits with renormalised softmax weights. A Switch-Transformer load-balancing loss
(`aux = E · Σᵢ fᵢ·Pᵢ`, weight 0.01) stops the router from collapsing onto one expert. The
baseline uses the same encoder with a single linear head.

**Class-balanced focal loss** ([`losses.py`](src/toxic_moe/losses.py)). Inverse-frequency
weighting (`#neg/#pos`) gives `threat` a positive weight of about 333, which makes the model
over-predict it. The *effective number of samples* `E(n) = (1−βⁿ)/(1−β)` from
[Cui et al., 2019](https://arxiv.org/abs/1901.05555) grows sub-linearly in `n`. Using it gives
a much softer weight `w_c = E(n⁻_c)/E(n⁺_c)` (about 24 for `threat` and 1.3 for `toxic` at
β = 0.9999). That weight is combined with focal modulation `(1−p_t)^γ`
([Lin et al., 2017](https://arxiv.org/abs/1708.02002)), γ = 2, which down-weights the flood
of easy negatives:

$$\ell_c = -\,w_c\,y\,(1-p)^{\gamma}\log p \;-\; (1-y)\,p^{\gamma}\log(1-p)$$

**Evaluation protocol** ([`experiment.py`](src/toxic_moe/experiment.py)):

| Split | Source | Used for |
|---|---|---|
| train | 90% of `train.csv`, stratified by label combination | gradient updates |
| validation | 10% of `train.csv` | best-epoch selection (ROC-AUC) **and** per-label F1 thresholds |
| test | Kaggle `test.csv` + `test_labels.csv`, unscored (`-1`) rows dropped → 63,978 comments | final numbers only, never used for any choice |

**Ablation** ([`configs/ablation.yaml`](configs/ablation.yaml)): {linear head, MoE head} × {BCE,
weighted BCE, focal, class-balanced focal}, each trained with **3 seeds** (42, 43, 44): 24 runs.
The data split is the same for every run, so seeds only change initialisation, dropout and batch
order. Tables report mean ± std over seeds, and the headline comparison is **paired by seed**:
on this dataset, differences between methods can be as small as seed-to-seed noise, and a single
run per setting cannot separate the two. All runs share the hyper-parameters: 2 epochs, AdamW,
lr 2e-5 for the encoder and 1e-4 for the head, 6% warm-up then linear decay, batch 32, max 128
tokens.

Engineering details:

* **Mixed precision:** bf16 on A100/L4, fp16 with loss scaling on older GPUs. DeBERTa-v3 runs
  in full precision (TF32 matmuls), because it produced NaN losses under bf16.
* **NaN guard:** a NaN/inf loss or gradient is never applied, and a run that keeps producing
  them stops after 20 bad steps instead of silently training on garbage.
* **Length-grouped batching with dynamic padding.** Most comments are far shorter than 128
  tokens, so this roughly halves the compute. It also keeps long inputs cheap: only the long
  comments get longer, so 512 tokens costs under 10% more time per run than 128.
* **Cached tokenisation.**
* **Resumable runs.** A finished run is skipped, so a Colab disconnect costs at most one run.
  Runs are ordered seed-major, so a partly finished study is still a complete single-seed ablation.
* **Fully deep-copied best checkpoint.**
* **CPU test suite with a tiny randomly initialised BERT,** run on every push by GitHub Actions.

## Reproduce

**Colab (recommended; A100 or L4 GPU).** Open
[`notebooks/train_on_colab.ipynb`](notebooks/train_on_colab.ipynb) with the badge above. It
runs the tests, loads the data (the zip from Kaggle's *Download All* button, uploaded to Google
Drive), trains the ablation (results go to Google Drive, so the notebook can resume after a
disconnect), builds the report, and lets you try the model on your own sentences. The 24 runs
took about 8 minutes each on an A100 (bf16).

**Local / any GPU machine:**

```bash
git clone https://github.com/qpp112/toxic-comment-moe && cd toxic-comment-moe
pip install -e ".[dev,deberta]"
pytest                                   # ~1 min on CPU, no downloads needed
# data: put the Kaggle "Download All" zip in data/ (it is unzipped automatically),
# or use the Kaggle API: bash scripts/download_data.sh data
python scripts/run_ablation.py --list    # show the 24 runs
python scripts/run_ablation.py --only moe_cbfocal --seeds 42 --set train_fraction=0.01 epochs=1 output_dir=results_smoke
python scripts/run_ablation.py                                   # main study -> results/<run>/metrics.json
python scripts/run_ablation.py --config configs/encoders.yaml    # optional DeBERTa-v3 study
python scripts/run_ablation.py --config configs/heads.yaml       # follow-up heads (docs/diagnosis.md)
python scripts/run_ablation.py --config configs/probes.yaml      # the same heads on a frozen encoder
python scripts/run_ablation.py --config configs/length.yaml      # longer inputs: 256 / 512 tokens, head + tail, control
python scripts/make_report.py --update-readme
python scripts/diagnose.py --data data                           # diagnostics (CPU) -> results/diagnosis.md
```

**Use a trained model:**

```bash
python -m toxic_moe.predict --run-dir results/moe_cbfocal_s42 "Thanks for fixing the citation!" "I will find you"
```

The output contains per-label probabilities and the labels that cross the validation-tuned
thresholds. Depending on the head, it also contains the router weights over the six experts
(MoE), each label's gate weights (multi-gate MoE), or the words each label attended to most
(label attention).

## Repository layout

```
src/toxic_moe/
  data.py         loading, test-label merge, stratified splits
  batching.py     cached tokenisation, dynamic padding, length-grouped batch sampler
  models.py       BERT encoder + linear / MLP / MoE / multi-gate MoE / label-attention head
  losses.py       BCE, weighted BCE, focal, class-balanced focal
  metrics.py      ROC-AUC, PR-AUC, F1 with validation-tuned thresholds, router statistics
  train.py        AMP training loop, warm-up/decay schedule, best-epoch selection
  experiment.py   one end-to-end run -> results/<name>/
  predict.py      inference on raw text
configs/          base settings, main ablation, DeBERTa study, follow-up heads, frozen-encoder probes, input length
scripts/          run_ablation.py, make_report.py, diagnose.py, download_data.sh
notebooks/        train_on_colab.ipynb
tests/            CPU test suite (tiny random BERT, synthetic data)
docs/             diagnosis of the MoE result, course-project write-up, figures
```

## Credits

The original course project was a team effort in UC Irvine CS178 (Machine Learning & Data
Mining), Fall 2025, Team 92: **Qui Phu Pham**, **Kat Strekalova** and **Tsunami Tazawa**.
Qui built the BERT models, the weighted-loss and threshold experiments and the first MoE
model. Kat handled data cleaning and the DistilBERT experiments. Tsunami built the Naive Bayes
baseline and the BiLSTM model. This repository (v2) is Qui's rewrite and extension of the
BERT/MoE part. It is new code, trained on the full dataset.

## References

* Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
* Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
* Cui et al. *Class-Balanced Loss Based on Effective Number of Samples.* CVPR 2019.
* Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017.
* Fedus, Zoph & Shazeer. *Switch Transformers.* JMLR 2022.
* Ma et al. *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts.* KDD 2018.
* Vu, Nguyen & Nguyen. *A Label Attention Model for ICD Coding from Clinical Text.* IJCAI 2020.
* Komatsuzaki et al. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR 2023.
* Jigsaw / Conversation AI. *Toxic Comment Classification Challenge.* Kaggle, 2018.

## License

Code: MIT (see [LICENSE](LICENSE)). The Jigsaw data is **not** included. Download it from
Kaggle under the competition's terms. The data contains offensive language.
