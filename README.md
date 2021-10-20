# Contents

* [Acknowledgement](#acknowledgement)
* [Citing This Paper](#citing-this-paper)
* [Environment Setup](#environment-setup)
* [Running Framework](#running-framework) 
* [Model Zoo](#model-zoo)


## Acknowledgement 
This repository contains the source code for the Revisiting HSIC Bottleneck for Adversarial Robustness project developed by the Northeastern University's SPIRAL research group. This research was generously supported by the National Science Foundation (grant CCF-1937500). 


## Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> Z. Wang*, T. Jian*, A. Masoomi, S. Ioannidis, J. Dy, "Revisiting Hilbert-Schmidt Information Bottleneck for Adversarial Robustness", NeurIPS, 2021.


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

You could produce the results of Figure 2 & 3, Table 1, 2 & 3 (PGD/HBaR+PGD) by this repository. Regarding MART and TRADES experiments (TRADES/MART/HBaR+TRADES/HBaR+MART), to make a fair comprision, we build our HSIC loss computation upon on MART's framework; but you can still run these experiments using our framework, which releases MART and TRADES loss in the function of 'mart_loss' and 'trades_loss' in ./source/hbar/core/train_misc.py.

To reproduce the HBaR experiments that we have in the paper, one could run our batch script by the following instruction:
    ```bash
    robust-mnist.sh     # HBaR training (HBaR-high) on MNIST 
    robust-cifar.sh     # HBaR training (HBaR-high) on CIFAR-10
    robust-mnist-adv.sh # Combining HBaR with adversarial learning on MNIST: HBaR+PGD
    robust-cifar-adv.sh # Combining HBaR with adversarial learning on CIFAR-10: HBaR+PGD
    ```
    
## Model Zoo

We are releasing selected models trained by HBaR for all three datasets used in the paper. Note that the numbers from these saved weights might differ a little bit from the ones reported in the paper.
