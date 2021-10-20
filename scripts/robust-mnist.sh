#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# basic exp:
# xw * xentropy + lx * hsic_xz - ly * hsic_yz
dataset=mnist
model=lenet3
xw=1
lx=1
ly=50

run_hbar -cfg config/general-hbar-xentropy-${dataset}.yaml -slmo -xw $xw -lx ${lx} -ly ${ly} \
    -mf ${dataset}_${model}_xw_${xw}_lx_${lx}_ly_${ly}.pt

