#!/bin/bash

#PBS -N defect_detection
#PBS -l select=1:ncpus=1:mem=32G:ngpus=2
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o out-run-train2.txt
#PBS -q normal

module load python/3.10.9
python3 ~/scratch/AirplaneDefectDetection/train_last_tuned_YOLOv8.py
