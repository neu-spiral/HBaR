#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset=cifar10
model=resnet18

xw=1
lx=0.006
ly=0.05

run_hbar -cfg config/general-hbar-xentropy-${dataset}.yaml -slmo -xw $xw -lx ${lx} -ly ${ly} -sd 444 \
    -mf ${dataset}_${model}_xw_${xw}_lx_${lx}_ly_${ly}.pt
