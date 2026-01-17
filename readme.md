# Minority-Class-aware Online Group Pruning Method for Image Segmentation Networks

## Description

We propose a minority-class-aware online group pruning method. Our method identifies and preserves
the most discriminative filters specific to minority classes, ensuring that the pruned network retains
its ability to accurately segment these underrepresented categories.

## Requirements

-  Python 3.9
-  Pytorch = 1.13.0
-  CUDA = 11.4

## Code Running

Datasets can be downloaded at [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and [Cityscapes](https://www.cityscapes-dataset.com/).

To reproduce our experiments, please use the following command:

### Pre-training
```shell
python prune_net_using_mcag_framework.py \
--gpus 0 \
--model UNet1 (or SegNet) \
--dataset camvid (or cityscapes) \
--classes 11 \
--max_epochs 300 \
--finetune_epochs 50 \
--pretrain \
--compress_rate 1 
```
### Pruning
```shell
python prune_net_using_mcag_framework.py \
--gpus 0 \
--model UNet1 (or SegNet) \
--dataset camvid (or cityscapes) \
--classes 11 \
--method Taylor Small Group \
--decay_rate 0.7 \
--max_epochs 300 \
--finetune_epochs 50 \
--pretrain [PRETRAIN_MODEL_PATH] \
--compress_rate 0.5 
```

## Comparative experiments
The following uses the SFP method as an example.
### Pre-training
```shell
python prune_net_using_sfp_framework.py (or \
--gpus 0 \
--model UNet1 (or SegNet) \
--dataset camvid (or cityscapes) \
--classes 11 (19 for cityscapes and 11 for camvid) \
--max_epochs 300 \
--finetune_epochs 50 \
--pretrain \
--compress_rate 1 
```
### Pruning
```shell
python prune_net_using_mcag_framework.py \
--gpus 0 \
--model UNet1 (or SegNet) \
--dataset camvid (or cityscapes) \
--classes 11 (11 for camvid and 19 for cityscapes) \
--method SFP \
--max_epochs 300 \
--finetune_epochs 50 \
--pretrain [PRETRAIN_MODEL_PATH] \
--compress_rate 0.5 
```

### Arguments
```shell
optional arguments:
  --model               Model name.
  --dataset             Camvid or Cityscapes.
  --input_size          Input size of model. default:camvid(360,480),cityscapes(512,1024)
  --classes             The number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively.
  --test_small          Test small model without zeors. default:False
  --method              Pruning method. default:Taylor Small Group
  --decay_rate          Decay rate for do mask. default:0.7
  --max_epochs          The number of epochs. default:300
  --finetune_epochs     The number of epochs for finetune training after set params to zeros. default:50
  --loss                Training loss. default:CrossEntropyLoss
  --lr                  Learning rate. default:5e-4
  --optim               Select optimizer. default:adam
  --gpus                Select gpu_id to use. default:[0]
  --pretrain            Path of the pre-trained model. 
  ----compress_rate     Pruning rate for model.
```

## Visualization

### CAM
```shell
python GradCAM.py 
```

### Segmentation results visualization 
```shell
python visualization.py 
```








