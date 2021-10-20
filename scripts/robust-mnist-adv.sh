#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Combining HBaR with adversarial training 

dataset=mnist
model=lenet3
xw=1
lx=0.003
ly=0.001

run_hbar -cfg config/general-hbar-xentropy-${dataset}.yaml -slmo -xw $xw -lx ${lx} -ly ${ly} \
    -adv \
    -mf ${dataset}_${model}_xw_${xw}_lx_${lx}_ly_${ly}_adv.pt