# Data, Checkpoints, and Predictions

Large resources are distributed separately from the GitHub repository.

## Download links

| Resource | Contents | Download |
| --- | --- | --- |
| Four public crack datasets | DeepCrack, CamCrack789, CrackMap, and Crack500 packages, LGC-DRNet checkpoints, and test predictions | [Google Drive](https://drive.google.com/file/d/1JVmg-cMsdvWd1agkVuvaid12ry_RnHeE/view?usp=sharing) |
| Self-built millimeter-scale concrete crack dataset (530 images) | `fine_crack_530_dataset`, LGC-DRNet checkpoints, and test predictions | [Google Drive](https://drive.google.com/file/d/1MHNzDTdEARDdEnv-uKD6HWre3TZO1mWB/view?usp=sharing) |
| MobileNetV3-Large backbone checkpoint | `mobilenet_v3_large-8738ca79.pth` used for backbone initialization | [Google Drive](https://drive.google.com/file/d/1cFuA8YgZplghXCD5Ez5Fo635cpyCUlLX/view?usp=sharing) |

The self-built dataset contains 530 pixel-wise annotated 512 × 512 images collected from concrete bridge components and concrete specimens, with 450 training and 80 test images.

## Checkpoints and predictions

The download packages contain trained LGC-DRNet checkpoints and their corresponding test predictions. Reported LGC-DRNet results are means of three independent runs.

## Dataset placement

After extraction, use the following structure:

```text
dataset/
└── <dataset_name>/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

Supported public-dataset names:

```text
DeepCrack
CamCrack789
CrackMap
Crack500
```

Self-built-dataset name:

```text
fine_crack_530
```

Retain the train/test organization contained in each package. The training script does not create a random dataset split. Keep the supplied 450/80 split for the self-built dataset when reproducing the reported results.

Masks must be PNG files. Pixel value `0` denotes background and every nonzero value denotes crack. Each mask must have the same original dimensions as its paired image.

Default filename mapping:

```text
sample.jpg -> sample.png
```

CamCrack789 mapping:

```text
image-001.jpg -> target-001.png
```

If `labels` is absent, the loader accepts a directory named `masks`. When both exist, `labels` takes precedence.

## Backbone placement

Place the downloaded backbone checkpoint at:

```text
vit_checkpoint/mobilenet_v3_large-8738ca79.pth
```

Enable it with:

```bash
python train.py \
  --pretrained 1 \
  --backbone_weights ./vit_checkpoint/mobilenet_v3_large-8738ca79.pth
```

The checkpoint must contain the complete torchvision MobileNetV3-Large state dictionary. Random initialization with `--pretrained 0` may produce substantially different results.

## Rights and citation notice

The self-built dataset is governed by [DATA_LICENSE.md](DATA_LICENSE.md). DeepCrack, CamCrack789, CrackMap, and Crack500 remain subject to their original licenses, access conditions, and citation requirements. The convenience package does not replace those terms. Before redistributing public-dataset files, confirm that each original license permits redistribution; otherwise provide only official source links, trained weights, and derived predictions.

The MobileNetV3-Large checkpoint and all other third-party materials remain subject to their original terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
