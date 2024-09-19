#!/bin/bash

#PBS -N YOLOv5
#PBS -l select=1:ncpus=1:mem=8G:ngpus=1
#PBS -l walltime=00:01:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-test-YOLOv5.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai_2
cd scratch/
python test_YOLOv5.py
