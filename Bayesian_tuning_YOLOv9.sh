#!/bin/bash

#PBS -N YOLOv9_Bayesian_tuning
#PBS -l select=1:ncpus=1:mem=128G:ngpus=1
#PBS -l walltime=90:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-Bayesian-tuning-YOLOv9.txt
#PBS -q normal
#PBS -M christian.paryoto@gmail.com


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/Bayesian_tune_YOLOv9.py
