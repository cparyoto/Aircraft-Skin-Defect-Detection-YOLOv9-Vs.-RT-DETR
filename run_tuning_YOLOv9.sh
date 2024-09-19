#!/bin/bash

#PBS -N YOLOv9_tuning
#PBS -l select=1:ncpus=1:mem=128G:ngpus=4
#PBS -l walltime=72:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-tuning-YOLOv9.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai
cd scratch/
python tune_YOLOv9.py
