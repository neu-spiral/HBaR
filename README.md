# Contents

* [Acknowledgement](#acknowledgement)
* [Citing This Paper](#citing-this-paper)
* [Abstract](#abstract)
* [Environment Setup](#environment-setup)
* [Running Framework](#running-framework) 
* [Model Zoo](#model-zoo)


## Acknowledgement 
This repository contains the source code for the Revisiting HSIC Bottleneck for Adversarial Robustness project developed by the Northeastern University's SPIRAL research group. This research was generously supported by the National Science Foundation (grant CCF-1937500). 


## Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> Z. Wang*, T. Jian*, A. Masoomi, S. Ioannidis, J. Dy, "Revisiting Hilbert-Schmidt Information Bottleneck for Adversarial Robustness", NeurIPS, 2021.

## Abstract
We investigate the HSIC (Hilbert-Schmidt independence criterion) bottleneck as a regularizer for learning an adversarially robust deep neural network classifier. In addition to the usual cross-entropy loss, we add regularization terms for every intermediate output of the neural networks to ensure that the latent representations retain useful information for output prediction while reducing redundant information from the input. We show that the HSIC bottleneck enhances robustness to adversarial attacks both theoretically and experimentally. In particular, we prove that the HSIC bottleneck regularizer reduces the sensitivity of the classifier to adversarial examples. Our experiments on multiple benchmark datasets and architectures demonstrate that incorporating an HSIC bottleneck regularizer attains competitive natural accuracy and improves adversarial robustness, both with and without adversarial examples during training.

## Environment Setup
Please install the python dependencies and packages found below:
```bash
pytorch-1.6.0
torchvision-0.7.0
numpy-1.16.1
scipy-1.3.1
tqdm-4.33.0
yaml-0.1.7
```
Please setup environment using:
```bash
source env.sh
```

## Running Framework

You could produce the results of Figure 2 & 3, Table 1, 2 & 3 (PGD/HBaR+PGD) by this repository. Regarding MART and TRADES experiments (TRADES/MART/HBaR+TRADES/HBaR+MART), to make a fair comprision, we build our HSIC loss computation upon on MART's framework; but you can still run these experiments using our framework, that releases MART and TRADES loss in the function of `mart_loss` and `trades_loss` in [./source/hbar/core/train_misc.py](./source/hbar/core/train_misc.py).

To reproduce the HBaR experiments that we have in the paper, one could run our batch script by the following instruction:
```bash
robust-mnist.sh     # HBaR training (HBaR-high) on MNIST 
robust-cifar.sh     # HBaR training (HBaR-high) on CIFAR-10
robust-mnist-adv.sh # Combining HBaR with adversarial learning on MNIST: HBaR+PGD
robust-cifar-adv.sh # Combining HBaR with adversarial learning on CIFAR-10: HBaR+PGD
```
 
Please refer to [./bin/run_hbar](./run_hbar) for more usages. The arguments in the code are self-explanatory.

## Model Zoo

We are releasing selected models trained by HBaR for all three datasets used in the paper. Note that the numbers from these saved weights might differ a little bit from the ones reported in the paper.

### Without Adversarial Training

| Dataset | Architecture | Model |
| --- | --- | --- |
| MNIST | LeNet | HBaR|
| CIFAR-10 | ResNet-18 | HBaR |

### Adversarial Training
| Dataset | Architecture | Model |
| --- | --- | --- |
| MNIST | LeNet | HBaR + PGD |
| CIFAR-10 | ResNet-18 | HBaR + TRADES |
| CIFAR-10 | WRN-28-10 | HBaR + TRADES |
| CIFAR-100 | WRN-28-10 | HBaR + TRADES |
