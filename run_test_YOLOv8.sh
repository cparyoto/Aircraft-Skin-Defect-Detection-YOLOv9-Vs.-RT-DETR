#!/bin/bash

#PBS -N YOLOv8
#PBS -l select=1:ncpus=1:mem=8G
#PBS -l walltime=00:15:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-test-YOLOv8.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai
cd scratch/
python test_YOLOv8.py
