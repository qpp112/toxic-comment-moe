# Results

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
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/ablation_dark.png">
  <img alt="Each setting minus the BERT + BCE baseline, paired by seed" src="../docs/figures/ablation_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/heads_dark.png">
  <img alt="Follow-up heads minus the BERT + BCE baseline, paired by seed" src="../docs/figures/heads_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/length_dark.png">
  <img alt="Longer inputs minus the 128-token baseline, paired by seed" src="../docs/figures/length_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/probes_dark.png">
  <img alt="Head gains with a frozen vs a fine-tuned encoder" src="../docs/figures/probes_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/mmoe_gates_dark.png">
  <img alt="Which experts each label's gate uses in the multi-gate MoE head" src="../docs/figures/mmoe_gates_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/per_label_f1_dark.png">
  <img alt="Per-label F1, baseline vs MoE + CB-focal" src="../docs/figures/per_label_f1_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/pr_curves_dark.png">
  <img alt="Precision–recall curves on the test set" src="../docs/figures/pr_curves_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../docs/figures/expert_routing_dark.png">
  <img alt="Which experts the router uses for each label" src="../docs/figures/expert_routing_light.png">
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
