#!/bin/bash

#PBS -N YOLOv9
#PBS -l select=1:ncpus=1:mem=32G:ngpus=1
#PBS -l walltime=00:10:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-test-YOLOv9.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/test_YOLOv9.py
