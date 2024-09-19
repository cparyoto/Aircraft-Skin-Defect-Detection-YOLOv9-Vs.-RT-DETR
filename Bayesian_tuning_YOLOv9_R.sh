#!/bin/bash

#PBS -N YOLOv9_Bayesian_tuning
#PBS -l select=1:ncpus=1:mem=128G:ngpus=1
#PBS -l walltime=50:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-Bayesian-tuning-YOLOv9_R.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/Bayesian_tune_YOLOv9_R.py
