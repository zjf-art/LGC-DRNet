# Data and Weights

The datasets, trained LGC-DRNet checkpoints, test predictions, and MobileNetV3-Large backbone weights are hosted on Google Drive because these files are too large for the GitHub repository.

## Download Links

| Resource | Contents | Download |
| --- | --- | --- |
| `public_datasets` | DeepCrack, CamCrack789, CrackMap, and Crack500 datasets, LGC-DRNet checkpoints, and test predictions | [Google Drive](https://drive.google.com/drive/folders/1HMdaT2yQir5Am6P_GyLymeC-S4H2hqDH?usp=sharing) |
| `fine_crack_530_dataset` | Training and test data, LGC-DRNet checkpoints, and test predictions | [Google Drive](https://drive.google.com/drive/folders/1ngnk8ducJu-7LaxSz_DRy8YAmzAn0--l?usp=sharing) |
| MobileNetV3-Large backbone weights | `mobilenet_v3_large-8738ca79.pth` used to initialize the LGC-DRNet backbone | [Google Drive](https://drive.google.com/drive/folders/1fAQLsMvx7hgghhKZFGi-GWpYxUeQ6k84?usp=sharing) |

## Checkpoints

The trained LGC-DRNet checkpoints correspond to the following three random seeds:

- `1729`
- `2825`
- `3141`

Each dataset directory contains the checkpoints and test predictions associated with these three runs. The reported experimental results are the averages of the three runs.

## Dataset Structure

After downloading and extracting the datasets, organize each dataset as follows:

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

The supported public dataset names are:

```text
DeepCrack
CamCrack789
CrackMap
Crack500
```

The original training and test directories provided in the download packages should be retained. The training scripts do not randomly divide the datasets.

All segmentation masks must be PNG files. A pixel value of `0` represents the background, while every nonzero value represents a crack. Each mask must have the same original dimensions as its corresponding image.

The default image-to-mask mapping is:

```text
sample.jpg -> sample.png
```

For CamCrack789, the mapping is:

```text
image-001.jpg -> target-001.png
```

The dataset loader accepts a `masks` directory when a `labels` directory is absent. If both directories exist, `labels` is used.

## Backbone Initialization

Place the downloaded MobileNetV3-Large backbone weights at:

```text
vit_checkpoint/mobilenet_v3_large-8738ca79.pth
```

Enable backbone initialization using:

```bash
python train.py --pretrained 1 --backbone_weights ./vit_checkpoint/mobilenet_v3_large-8738ca79.pth
```

The file must contain the complete torchvision MobileNetV3-Large state dictionary. Using `--pretrained 0` results in random backbone initialization and may produce different experimental results.

## Public Dataset Notice

DeepCrack, CamCrack789, CrackMap, and Crack500 remain subject to their original licenses and terms of use. The datasets are provided for research and reproducibility purposes. Users should cite the corresponding original publications when using these datasets.
