# ACORN

Code for "A dNBR-Anchored Ordinal Network with Inferred Class Boundaries for Burn Severity Mapping".

ACORN scores every pixel with dNBR plus a learned residual and converts the score into four ordered severity classes with three
cut points. The cut points of each fire are inferred from the unlabeled imagery of the whole fire by a fire encoder, and an unseen
fire needs no reference labels or published thresholds.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.9 and segmentation-models-pytorch 0.5.0 on a CUDA GPU. ImageNet weights of the ResNet-34
encoder are downloaded on first use.

## Data

Two benchmarks are distributed as zip files on [Google Drive](https://drive.google.com/drive/folders/1An4IAtOCSruUWXanfcH5qMkRmKAvmnLt?usp=sharing).

| File | Contents |
|---|---|
| `ACORN_HLS.zip` | 2,524 MTBS fires and 120 Copernicus EMS activations with HLS Landsat and Sentinel-2 imagery, 6,966 labelled tiles |
| `ACORN_BAER.zip` | 795 BAER soil burn severity fires, 4,649 labelled tiles |

Each archive holds the pre-fire and post-fire scenes, the reference labels, the tile table and the five spatial folds. See the
README inside each archive for the file formats.

## Preparing the caches

`prepare.py` cuts the scenes into the labelled 256 x 256 tiles used for training and evaluation (`patches`) and into the
label-independent tiles that cover each fire for the fire encoder (`context`).

```bash
unzip ACORN_HLS.zip -d data
unzip ACORN_BAER.zip -d data
python prepare.py --dataset hls --root data/ACORN_HLS --out cache/hls
python prepare.py --dataset baer --root data/ACORN_BAER --out cache/baer
```

The HLS caches take about 12 GB (patches) and 22 GB (context), the BAER caches 8 GB and 24 GB.

## Training and testing

For test fold k, fold (k + 1) mod 5 selects the epoch and the remaining three folds are used for training. The defaults are the
settings of the paper.

```bash
for fold in 0 1 2 3 4; do
  for seed in 42 123 777; do
    python train.py --data cache/hls/patches --context cache/hls/context --fold $fold --seed $seed --out runs/hls/f${fold}_s${seed}
    python train.py --data cache/baer/patches --context cache/baer/context --fold $fold --seed $seed --out runs/baer/f${fold}_s${seed}
  done
done
```

Each run writes `best.pt` and `result.json`, which holds per-fire confusion matrices of the test fold for ACORN (`test_fire_cm`),
for ACORN with the base boundaries of all fires (`z0_fire_cm`) and for the fixed USGS thresholds (`usgs_fire_cm`), together with
the inferred latent of every test fire.

Cross-sensor transfer trains and validates on one HLS sensor and tests on the other.

```bash
python train.py --data cache/hls/patches --context cache/hls/context --fold 0 --seed 42 --sensors landsat --test_sensors s2 --out runs/l2s/f0_s42
```

A trained checkpoint can be evaluated again with `test.py`.

```bash
python test.py --checkpoint runs/hls/f0_s42/best.pt --data cache/hls/patches --context cache/hls/context --out runs/hls/f0_s42/test.json
```

## Evaluation

`evaluate.py` pools the test predictions of the five folds of each seed into one confusion matrix and reports mIoU and FWIoU
averaged over seeds, with the standard deviation across seeds, and the class IoU.

```bash
python evaluate.py "runs/hls/*" --program mtbs
python evaluate.py "runs/hls/*" --program ems
python evaluate.py "runs/baer/*"
python evaluate.py "runs/hls/*" --program mtbs --sensor landsat
python evaluate.py "runs/hls/*" --program mtbs --field z0_fire_cm
```

## License

The code is released under the [MIT License](LICENSE). The imagery and reference labels in the data archives remain subject to the
terms of their original providers (HLS, MTBS, Copernicus EMS and BAER).
