#!/bin/bash

#PBS -N YOLOv9_GBRT_tuning
#PBS -l select=1:ncpus=1:mem=32G:ngpus=1
#PBS -l walltime=40:00:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-GBRT-tuning-YOLOv9.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai_2
python ~/scratch/GBRT_tune_YOLOv9.py
