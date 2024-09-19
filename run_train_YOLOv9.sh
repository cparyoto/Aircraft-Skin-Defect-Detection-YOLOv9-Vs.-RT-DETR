#!/bin/bash

#PBS -N YOLOv9
#PBS -l select=1:ncpus=1:mem=128G:ngpus=4
#PBS -l walltime=24:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-train-YOLOv9.txt
#PBS -q normal
#PBS -M christian.paryoto@gmail.com


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/train_Bayesian_tuned_YOLOv9.py
