#!/bin/bash

#PBS -N RT-DETR
#PBS -l select=1:ncpus=1:mem=128G:ngpus=2
#PBS -l walltime=24:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-train-RT-DETR.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/train_Bayesian_tuned_RTDETR_150epochs.py
