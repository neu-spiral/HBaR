#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Combining HBaR with adversarial training 

dataset=cifar10
model=resnet18

xw=1
lx=0.0005
ly=0.005

run_hbar -cfg config/general-hbar-xentropy-${dataset}.yaml -slmo -xw $xw -lx ${lx} -ly ${ly} \
-adv -ep 83 \
-mf ${dataset}_${model}_xw_${xw}_lx_${lx}_ly_${ly}_adv.pt