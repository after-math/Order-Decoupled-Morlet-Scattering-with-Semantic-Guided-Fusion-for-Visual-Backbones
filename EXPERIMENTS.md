# Main-text code map

This release implements the static method. No reported accuracies, trained weights or generated plots are shipped.

| Main-text item | Scope | Entry points |
| --- | --- | --- |
| Order configurations | Flowers102, RSSCN7; seven backbones; FixedBanks, Shared, FullStatic; D=128 | `tools/plan.py orders`, `model.py` |
| Gate-context mismatches | Same two datasets and seven backbones; own, same-class, other-class, random; three donor seeds | `tools/plan.py gates`, `interventions.py` |
| Learned filter and response figure | Static Shared versus FullStatic, matched second-order path | `tools/compare_orders.py` |
| Gate maps and class contributions | Own versus swapped semantic context in one frozen static model | `tools/visualize.py` |
| Limited-label comparison | Flowers102 k=5,10; AID/Flowers17 k=5,10,20,40; KTH-TIPS2-b k=5,10,20,30,40 across four folds; five backbones | `tools/plan.py limited`, `data.py` |
| Full-supervision comparison | AID, Flowers102, RSSCN7, four KTH-TIPS2-b folds; seven backbones | `tools/plan.py main`, `train.py` |
| Frozen ImageNet comparison | AID, RSSCN7; seven backbones; native classifier versus static augmentation; D=128 | `tools/plan.py frozen`, `full_frame.py`, `static_original.py` |
| Component ablation | Flowers102, RSSCN7; EdgeNeXt-Small and MobileViTv2-1.0; five controls; D=128 | `tools/plan.py components`, `model.py` |
| Parameter/FLOP table and latency | Native and FullStatic; seven backbones | `tools/benchmark.py`, `flop_counter.py` |
| New-run aggregation | Four-fold means and paired percentage-point gains | `tools/summarize.py` |

The five limited-label backbones are ShuffleNetV2-1.0 (D=64), EfficientNet-B0, EdgeNeXt-Small, MambaOut-Femto (D=128), and ConvNeXt-Tiny (D=256). MobileViTv2-1.0 remains available in the full-supervision and mechanism/component comparisons with D=128.

Caltech101 and limited-label RSSCN7 are supplementary-only and have no release task plans. UCMerced/SIRI-WHU frozen runs, frozen KTH extensions, filter transfer, robustness experiments and second-order probing are not included.

The historical manuscript's `Both random` condition exchanges both the gate context and a dynamic parameter-generator input. A static model has no such generator. That legacy condition and dynamic E4 checkpoint support are deliberately absent; they are not relabeled as static results. The code can generate new static comparisons, not reproduce dynamic checkpoint rows without retraining. This release does not edit the manuscript or its tables.
