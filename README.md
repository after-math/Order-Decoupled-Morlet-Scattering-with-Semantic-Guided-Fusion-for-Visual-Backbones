# Order-Decoupled Morlet Scattering

Code for the main-text experiments on static, order-decoupled Morlet scattering and semantic-guided residual fusion. The repository contains source, executable experiment plans and documentation only.

## Method

Two independent trainable Morlet banks are shared across images. Each bank has 18 kernels with four controls per kernel: orientation, carrier frequency, envelope width and anisotropy. The first-order modulus feeds 108 second-order paths. The 54 first-order and 324 second-order RGB channels are pooled and projected separately, concatenated, and combined into a width-D texture feature. A joint gate receives backbone context and texture features; its output weights the texture residual before addition to the native feature map. The residual scale starts at 0.1.

“Static” means that the learned filters do not depend on the current image. The gate remains image-dependent. There is no filter-parameter generator or image-conditioned filter offset in this implementation.

## Installation

Use Python 3.10 or 3.11 on Linux with CUDA. Install the matching PyTorch CUDA build, then:

```bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

CUDA uses the archived accelerated FFT/lowpass implementation, including optional Triton kernels and activation checkpointing. CPU uses the PyTorch reference path and is supported for verification and inference. The training runner requires CUDA.

## Data

Datasets and split manifests are not distributed. A manifest contains `dataset`, `classes`, `data_root`, and `splits`. Each of `train`, `validation`, and `test` is a list of `{"relative_path": "class/image.jpg", "label": 0}` records. Labels are zero-based indices into `classes`. Import the saved splits used by your experiment:

```bash
python tools/prepare_manifest.py --source /path/to/saved/manifest.json \
  --data-root /path/to/dataset --output inputs/RSSCN7.json
```

This validates file availability, labels, path containment and split separation; it copies no images or results. Keep the saved AID 4000/1000/5000 and RSSCN7 1960/420/420 partitions. Flowers102 uses the official train/validation/test partitions. Flowers17 uses its archived partition rather than a newly sampled split.

For KTH-TIPS2-b, either import the archived four manifests or generate the physical-specimen folds:

```bash
python tools/prepare_kth.py --data-root /path/to/KTH-TIPS2-b --output-dir inputs
```

The generator expects `category/sample_a` through `category/sample_d`, 11 categories, 108 images per specimen, and folds `(ab,c,d)`, `(bc,d,a)`, `(cd,a,b)`, `(da,b,c)`. Do not split individual views at random.

## Training

```bash
# Full supervision: run each variant with the same manifest and seed.
python train.py --manifest inputs/RSSCN7.json --backbone edgenext_small \
  --variant FullStatic --width 128 --output runs/rsscn7_edgenext_static
python train.py --manifest inputs/RSSCN7.json --backbone edgenext_small \
  --variant Native --width 128 --output runs/rsscn7_edgenext_native

# Limited labels: deterministic, nested per-class subsets from training only.
python train.py --manifest inputs/Flowers102.json --backbone mobilevitv2_100 \
  --variant FullStatic --width 128 --shots 5 --output runs/flowers_mv2_k5
```

Seven supported backbones: `edgenext_small`, `shufflenet_v2_x1_0`, `mobilevitv2_100`, `resnet18`, `convnext_tiny`, `efficientnet_b0`, `mambaout_femto`.

Default full-supervision/limited-label widths are 64 for ShuffleNetV2, 256 for ConvNeXt-Tiny, and 128 otherwise. Order, frozen-backbone and component comparisons explicitly use 128. MobileViTv2-1.0 uses 128 throughout.

The runner uses seed 42, batch 32, 256-pixel inputs, up to 200 epochs, patience 25, AdamW, ordinary LR 0.0004, Morlet LR 0.003, five warmup epochs, cosine decay, label smoothing 0.05, Mixup alpha 0.2 with probability 0.5, and gradient clipping at 1. Validation selects the first best checkpoint; the test set is evaluated after training. `--resume` restores model, optimizer, scaler, best state and history. Epoch random streams are regenerated deterministically.

## Main-text experiment plans

```bash
python tools/plan.py orders > /tmp/orders.sh
python tools/plan.py limited > /tmp/limited.sh
python tools/plan.py main > /tmp/main.sh
python tools/plan.py frozen > /tmp/frozen.sh
python tools/plan.py components > /tmp/components.sh
python tools/plan.py gates > /tmp/gates.sh
```

These commands print training commands and do not launch jobs. Edit data/weight destinations for your machine. The limited-label plan covers the four main-text datasets and the five backbones in the limited-label comparison; MobileViTv2 is also available through the training CLI. The frozen plan covers AID and RSSCN7. Supplement-only studies are excluded.

| Variant | Definition |
| --- | --- |
| Native | Unmodified native classifier |
| FullStatic | Independent trainable first- and second-order banks, joint gate, learned scale |
| Shared | One trainable bank, with indexed scale-transition sharing |
| FixedBanks | Both independent banks fixed at initialization; projections and fusion learned |
| FirstOrderOnly | Zero second-order projected readout |
| SecondOrderOnly | Zero first-order projected readout; retain first-order modulus to compute second order |
| NoGate | Direct texture residual, gate fixed to one |
| FixedAlpha | Residual scale fixed at 0.1 |

## Frozen ImageNet backbones

Obtain the local ImageNet weights listed in `configs/imagenet.json`. No weights are bundled. Use the normalization associated with each weight source; MobileViTv2's source uses RGB values without the ImageNet mean/std transform.

```bash
python train.py --manifest inputs/AID.json --backbone edgenext_small \
  --variant FullStatic --width 128 --frozen \
  --imagenet weights/edgenext_small.safetensors --output runs/aid_frozen_static
```

The frozen path keeps original pixels for scattering, and creates a long-edge-256, mean-padded native view. Content-coordinate sampling and coverage masks align the residual with the valid native image region. Frozen training uses horizontal flips and full-frame Mixup; microbatches accumulate into the same global batch of 32.

The loader requires every non-classifier body tensor to match. The task head retains its paired random initialization. For MambaOut, only `head.fc2` is the task classifier; its hidden head is preserved and frozen. All other native body parameters and running statistics are frozen; only the classifier and added branch are trained. Repeat with `--variant Native` for the classifier-only control.

## Gate-context interventions

```bash
python interventions.py --checkpoint runs/rsscn7_edgenext_static/best.pt \
  --manifest inputs/RSSCN7.json --output runs/interventions.json
```

Evaluate own, same-class, other-class and random gate context using frozen static filters. Maps are one-to-one, exclude self-pairs, and use seeds 42–44. Only the semantic context entering the gate is replaced; the target image, texture features and native path remain fixed. `Both random` from the archived dynamic model has no equivalent here because there is no generator to swap.

## Figures and computational cost

```bash
python tools/visualize.py --checkpoint runs/rsscn7_edgenext_static/best.pt \
  --manifest inputs/RSSCN7.json --sample 336 --output runs/mechanism.pdf
python tools/benchmark.py --backbone edgenext_small --width 128 --device cuda:0
python tools/compare_orders.py --shared runs/shared/best.pt --separate runs/static/best.pt \
  --manifest inputs/RSSCN7.json --sample 336 --output runs/order_comparison.pdf
```

The figure code shows static filter spectra, second-order responses, own/other gate maps and signed integrated-gradient contributions along the gated-feature interpolation. The displayed kernel is selected explicitly, and spectral contours compare sharing counterparts. It generates new visuals from the supplied checkpoint, without bundling article figures or selected-example outputs. FLOPs use MAC=2 and complex FFT cost 5P log2(P); latency includes warmup and CUDA synchronization.

## Scope and provenance

The numerical core was consolidated from the archived static two-bank component experiments, the optimized static runtime implementation and the main-text data/training code. Dynamic variants were replaced by the static implementation, not renamed in place. Supplementary-only counterfactual-generator, second-order probing, filter-transfer and robustness studies are not included.

This is a unified **static-method code release**. Historical article tables that used dynamic E4 checkpoints are not reproduced by pretending those checkpoints were static. New runs produce their own outputs. This folder contains no experimental results, checkpoints, dataset images, PDFs or supplemental material. The article itself and all archived results remain untouched.

See `EXPERIMENTS.md` for the complete main-text scope and the legacy-condition boundary.

See `THIRD_PARTY.md` for the bundled MambaOut implementation. No project-wide license has been selected on behalf of the author.
