#!/bin/bash

#PBS -N RTDETR_Bayesian_tuning_R
#PBS -l select=1:ncpus=1:mem=128G:ngpus=1
#PBS -l walltime=08:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-Bayesian-tuning-RTDETR_R.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/Bayesian_tune_RTDETR_150epochs_R.py
