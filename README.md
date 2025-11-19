# Group 27: Feature Pyramid Network - Modern PyTorch Implementation

A modernized implementation of [FPN](https://arxiv.org/pdf/1612.03144.pdf) in PyTorch, based on [easy-fpn.pytorch](https://github.com/potterhsu/easy-fpn.pytorch). Updated for PyTorch 2.x with native torchvision operations, eliminating the need for custom CUDA compilation.

## Key Differences from Original

**PyTorch Compatibility:**
- Updated from PyTorch 0.4.1 to PyTorch 2.x
- Replaced deprecated `torch.utils.ffi` with modern APIs
- Removed legacy CUDA extension dependencies

**Simplified Dependencies:**
- Uses `torchvision.ops.nms` instead of custom CUDA NMS
- Uses `torchvision.ops.roi_align` instead of custom CUDA ROI Align
- No manual CUDA compilation required

**Bug Fixes:**
- Fixed P6 generation: corrected `kernel_size=1` to `kernel_size=2` in max pooling
- Fixed device placement issues for CUDA/CPU tensor operations

**Performance:**
- Achieves **75.39% mAP** on PASCAL VOC 2007 (reference: 76%)
- Fully functional with modern PyTorch 2.x and torchvision

## Demo

![](images/inference-result.jpg?raw=true)


## Features

* Supports PyTorch 2.x with CUDA
* Supports `PASCAL VOC 2007` and `MS COCO 2017` datasets
* Supports `ResNet-18`, `ResNet-50` and `ResNet-101` backbones (from torchvision)
* Supports `ROI Pooling` and `ROI Align` pooling modes
* Clean, maintainable code with modern PyTorch practices


## Benchmarking

* PASCAL VOC 2007

    * Train: 2007 trainval (5011 images)
    * Eval: 2007 test (4952 images)

    | Implementation | Backbone | GPU | mAP | Training Steps |
    |---|---|---|---|---|
    | Original ([easy-fpn.pytorch](https://github.com/potterhsu/easy-fpn.pytorch)) | ResNet-101 | GTX 1080 Ti | 0.7627 | 70000 |
    | **Ours (Modern)** | ResNet-101 | RTX 4080 SUPER | **0.7539** | 80000 |

    **Per-class AP:**
    ```
    aeroplane: 79.46%    bicycle: 85.22%      bird: 77.73%         boat: 62.39%
    bottle: 65.08%       bus: 82.30%          car: 86.65%          cat: 86.26%
    chair: 59.08%        cow: 83.02%          diningtable: 67.86%  dog: 86.17%
    horse: 84.00%        motorbike: 78.72%    person: 79.28%       pottedplant: 51.09%
    sheep: 75.62%        sofa: 65.10%         train: 77.09%        tvmonitor: 75.58%
    ```


## Requirements

* Python 3.6+
* PyTorch 2.x with CUDA support
* torchvision (compatible with PyTorch version)
* tqdm
* tensorboardX

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install tqdm tensorboardX
    ```


## Setup

1. **Prepare data**

    For `PASCAL VOC 2007`:

    1. Download dataset
        - [Training / Validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (5011 images)
        - [Test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (4952 images)

    2. Extract to data folder:
        ```
        feature-pyramid-networks-project/
            data/
                VOCdevkit/
                    VOC2007/
                        Annotations/
                        ImageSets/Main/
                        JPEGImages/
        ```

    For `MS COCO 2017`:

    1. Download dataset
        - [2017 Train images [18GB]](http://images.cocodataset.org/zips/train2017.zip)
        - [2017 Val images [1GB]](http://images.cocodataset.org/zips/val2017.zip)
        - [2017 Annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

    2. Extract to data folder:
        ```
        feature-pyramid-networks-project/
            data/
                COCO/
                    annotations/
                    train2017/
                    val2017/
        ```

2. **Install pycocotools** (for MS COCO 2017 only)

    ```bash
    pip install pycocotools
    ```


## Usage

1. **Train**

    ```bash
    # PASCAL VOC 2007 with ResNet-101
    python train.py -s=voc2007 -b=resnet101
    
    # MS COCO 2017 with custom pooling mode
    python train.py -s=coco2017 -b=resnet101 --pooling_mode=align
    ```

2. **Evaluate**

    ```bash
    # Evaluate on VOC 2007
    python eval.py -s=voc2007 -b=resnet101 outputs/checkpoints-xxx/model-80000.pth
    
    # Evaluate on COCO 2017
    python eval.py -s=coco2017 -b=resnet101 outputs/checkpoints-xxx/model-80000.pth
    ```

3. **Inference**

    ```bash
    # Run inference on single image
    python infer.py -c=outputs/checkpoints-xxx/model-80000.pth \
        -s=voc2007 -b=resnet101 \
        input.jpg output.jpg
    
    # With custom probability threshold
    python infer.py -c=outputs/checkpoints-xxx/model-80000.pth \
        -s=voc2007 -b=resnet101 -p=0.9 \
        input.jpg output.jpg
    ```


## Notes

* **Feature Pyramid Architecture** (see `forward` in `model.py`):

    ```python
    # Bottom-up pathway
    c1 = self.conv1(image)
    c2 = self.conv2(c1)
    c3 = self.conv3(c2)
    c4 = self.conv4(c3)
    c5 = self.conv5(c4)

    # Top-down pathway and lateral connections
    p5 = self.lateral_c5(c5)
    p4 = self.lateral_c4(c4) + F.interpolate(input=p5, size=(c4.shape[2], c4.shape[3]), mode='nearest')
    p3 = self.lateral_c3(c3) + F.interpolate(input=p4, size=(c3.shape[2], c3.shape[3]), mode='nearest')
    p2 = self.lateral_c2(c2) + F.interpolate(input=p3, size=(c2.shape[2], c2.shape[3]), mode='nearest')

    # Reduce the aliasing effect
    p4 = self.dealiasing_p4(p4)
    p3 = self.dealiasing_p3(p3)
    p2 = self.dealiasing_p2(p2)

    # Fixed: kernel_size=2 (was incorrectly 1 in some implementations)
    p6 = F.max_pool2d(input=p5, kernel_size=2, stride=2)
    ```

    ![](images/feature-pyramid.png)

* **Modern PyTorch Operations:**
    - NMS: Uses `torchvision.ops.nms` (no custom CUDA required)
    - ROI Align: Uses `torchvision.ops.roi_align` (no custom CUDA required)
    - Fully compatible with PyTorch 2.x


## Acknowledgments

Based on [easy-fpn.pytorch](https://github.com/potterhsu/easy-fpn.pytorch) by potterhsu. Modernized for PyTorch 2.x with bug fixes and simplified dependencies.
