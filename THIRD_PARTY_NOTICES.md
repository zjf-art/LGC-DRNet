# Third-Party Notices

The BSD 3-Clause License in this repository applies only to original LGC-DRNet source code for which the repository authors hold the necessary rights. It does not relicense third-party software, datasets, pretrained weights, publications, or figures.

## PyTorch and torchvision

LGC-DRNet depends on PyTorch and torchvision and uses `torchvision.models.mobilenet_v3_large` and `torchvision.ops.DeformConv2d`. PyTorch and torchvision are third-party projects distributed under their own licenses. Users are responsible for reviewing and complying with the license terms distributed by those projects.

## MobileNetV3-Large backbone checkpoint

The file `mobilenet_v3_large-8738ca79.pth` is distributed separately and is not covered by this repository's BSD 3-Clause License. Its use remains subject to the applicable upstream terms. The checkpoint is provided only as the initialization resource used by the experiments.

## Public crack datasets

DeepCrack, CamCrack789, CrackMap, and Crack500 are third-party datasets. Their images, labels, names, and associated documentation remain subject to the original owners' licenses, terms of use, and citation requirements. Inclusion of a download link or convenience archive does not transfer ownership or grant additional permissions.

Users must cite the corresponding original publications and obtain any permission required for their intended use. If an original dataset license does not permit redistribution, the repository maintainers should distribute only official source links and original LGC-DRNet-derived artifacts such as checkpoints and predictions.

## Comparison results and figures

Some quantitative comparison values are taken from Ref. [22] in the manuscript and are identified as literature results in the README and paper. Comparison-model names and results are provided for scholarly reference only; implementations of those models are not included.

## Camera and deployment materials

Camera CAD/3D models, circuit and assembly materials, RKNN files, and RV1106 deployment code are not included in the current release. After the related invention-patent applications receive official application numbers and acceptance notices, the authors plan to supplement this repository with camera-system 3D model files and technical materials for edge deployment. Any applicable license terms and third-party notices will be provided with those materials when released.
