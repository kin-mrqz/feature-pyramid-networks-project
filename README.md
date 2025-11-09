# Feature Pyramid Networks (FPN) Reproduction

## Paper: Feature Pyramid Networks for Object Detection

**Authors**: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie  
**Conference**: CVPR 2017

This is a repository containing our implementation of the Feature Pyramid Network for Object Detection and Classification as part of our course requirement for COMP3314: Introduction to Machine Learning

## Setup

## Running ...

Detectron2:

- tldr
- implementation: https://github.com/facebookresearch/detectron2

Architecture: `"COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"`

- Faster R-CNN detection framework
- ResNet-50 backbone
- Feature Pyramid Network
- 1x training schedule (90,000 iterations)
  Detectron2's built-in FPN from their model zoo

## Reproduction Details:

1. download datasets:

## Results:

...

Differences due to:

1. custom split (115k train) vs standard coco 2017 split (118k train + 5k val)

2. different resnet implementations: original has been deprecated
3. hypermarameters (learning rate, batch size)
4. hardware differences (optimization, etc.)

## Difficulties Encountered

- gpu session time limits + checkpoint issues
