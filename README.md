<img src=/images/logo_github.png  width="30%" height="15%">

# Towards training digitally tied analog blocks via hybrid gradient computation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<p float="left">
<img src=/images/main_fig.png  width="40%" height="20%">
<img src=/images/block_fwd_bwd.png  width="50%" height="40%">
</p>



We release in this repo the codebase which was used to produce the experimental results of ["Towards training digitally-tied analog blocks via hybrid gradient computation" (Nest & Ernoult, 2024)](https://arxiv.org/abs/2409.03306).

## Pre-requisites
This codebase requires:
- **Python** >= 3.10
- **Pytorch** (select [here](https://pytorch.org) depending on your preferences). The version of the Pytorch packages we used are the following ones:
  ```
  torch==2.0.1
  torchaudio==2.0.2
  torchmetrics==1.3.2
  torchvision==0.15.2
  ```
- **Hydra** for terminal prompt and configuration files parsing (see [here](https://hydra.cc/docs/intro/) for installation details).
- **gdown and wget** to download the ImageNet32 (ImageNet 1k dataset already downsampled to 32x32 pixels):
  ```
  pip install gdown
  pip install wget
  ```


- Optionally: **Tensorboard** or **wandb** to track training experiments.


## Structure of the codebase
The codebase manipulates three independent types:
- `Algorithm`: either backpropagation through time (`BPTT`) -- which boils down to Implicit Differentiation (ID) in the context of equilibrium models -- or equilibrium propagation (`EP`).
- `HopfieldChain`: models are of type `HopfieldChain`, which themselves read as compositions of `HopfieldBlock` objects. There are two subtypes of `HopfieldChain`: `VGG` and `ResNet` types. `VGG` and `ResNet` are chains of `ConvPool` and `BasicBlock` objects, which are themselves subtypes of `HopfieldBlock`.
- `Experiment`: there are training (`TrainingExperiment`) and static gradient analysis (`GDDExperiment`) experiments.


## Reproducing Experiments
The codebase utilizes hydra to modularize experiments. The following commands can be used to reproduce results reported in the paper:

### Static gradient analysis experiments
This pertains to Figures 3 and 4 of the [initial ArXiV release of our paper](https://arxiv.org/abs/2409.03306) and Figures 3 and 5 of the camera-ready version (soon to be released).

To reproduce these figures, hit the following command:

```
python main.py --config-name=gdd 
```


### Splitting experiments with a *convergence criterion* (TOL)
 
This is Table 1 of the camera-ready version (soon to be released). This table was not included in the [initial ArXiV release of our paper](https://arxiv.org/abs/2409.03306). 

|L |bs |Algorithm | Command|
|:------:|:------:|:------:|:--------------------------|
|6|6| EP | `python main.py --config-name=splitting_small_TOL model=splitting_small_1block`
|6|6| ID | `python main.py --config-name=splitting_small_TOL model=splitting_small_1block algorithm=bptt`
|6|3| EP | `python main.py --config-name=splitting_small_TOL model=splitting_small_2block`
|6|3| ID | `python main.py --config-name=splitting_small_TOL model=splitting_small_2block algorithm=bptt`
|6|2| EP | `python main.py --config-name=splitting_small_TOL model=splitting_small_3block`
|6|2| ID | `python main.py --config-name=splitting_small_TOL model=splitting_small_3block algorithm=bptt`
|12|4| EP | `python main.py --config-name=splitting_large_TOL model=splitting_large_3block`
|12|4| ID | `python main.py --config-name=splitting_large_TOL model=splitting_large_3block algorithm=bptt`
|12|3| EP | `python main.py --config-name=splitting_large_TOL model=splitting_large_4block`
|12|3| ID | `python main.py --config-name=splitting_large_TOL model=splitting_large_4block algorithm=bptt`
|12|2| EP | `python main.py --config-name=splitting_large_TOL model=splitting_large_6block`
|12|2| ID | `python main.py --config-name=splitting_large_TOL model=splitting_large_6block algorithm=bptt`


### Splitting experiments with a *fixed number of iterations* 

This is Table 1 of the [initial ArXiV release of our paper](https://arxiv.org/abs/2409.03306) and Table 7 of the camera-ready version (soon to be released). 

|L |bs |Algorithm | Command|
|:------:|:------:|:------:|:--------------------------|
|6|6| EP | `python main.py --config-name=splitting_small model=splitting_small_1block`
|6|6| ID | `python main.py --config-name=splitting_small model=splitting_small_1block algorithm=bptt`
|6|3| EP | `python main.py --config-name=splitting_small model=splitting_small_2block`
|6|3| ID | `python main.py --config-name=splitting_small model=splitting_small_2block algorithm=bptt`
|6|2| EP | `python main.py --config-name=splitting_small model=splitting_small_3block`
|6|2| ID | `python main.py --config-name=splitting_small model=splitting_small_3block algorithm=bptt`
|12|4| EP | `python main.py --config-name=splitting_large model=splitting_large_3block`
|12|4| ID | `python main.py --config-name=splitting_large model=splitting_large_3block algorithm=bptt`
|12|3| EP | `python main.py --config-name=splitting_large model=splitting_large_4block`
|12|3| ID | `python main.py --config-name=splitting_large model=splitting_large_4block algorithm=bptt`
|12|2| EP | `python main.py --config-name=splitting_large model=splitting_large_6block`
|12|2| ID | `python main.py --config-name=splitting_large model=splitting_large_6block algorithm=bptt`


### Scaling experiments (Table 2)

|L |Dataset |Algorithm | Command|
|:------:|:------:|:------:|:--------------------------|
12|CIFAR100|EP|`python main.py --config-name=scaling_small model=scaling_small data=cifar100`
12|CIFAR100|ID|`python main.py --config-name=scaling_small model=scaling_small data=cifar100 algorithm=bptt`
12|ImageNet32|EP|`python main.py --config-name=scaling_small model=scaling_small data=imagenet32`
12|ImageNet32|ID|`python main.py --config-name=scaling_small model=scaling_small data=imagenet32 algorithm=bptt`
15|CIFAR100|EP|`python main.py --config-name=scaling_large model=scaling_large data=cifar100`
15|CIFAR100|ID|`python main.py --config-name=scaling_large model=scaling_large data=cifar100 algorithm=bptt`
15|ImageNet32|EP|`python main.py --config-name=scaling_large model=scaling_large data=imagenet32`
15|ImageNet32|ID|`python main.py --config-name=scaling_large model=scaling_large data=imagenet32 algorithm=bptt`
