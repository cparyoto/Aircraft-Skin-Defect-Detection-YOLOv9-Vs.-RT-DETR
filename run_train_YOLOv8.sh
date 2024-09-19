#!/bin/bash

#PBS -N YOLOv8
#PBS -l select=1:ncpus=1:mem=128G:ngpus=2
#PBS -l walltime=03:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-train-YOLOv8.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai
cd scratch/
python train_YOLOv8.py
