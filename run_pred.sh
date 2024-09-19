#!/bin/bash

#PBS -N pred
#PBS -l select=1:ncpus=1:mem=32G:ngpus=1
#PBS -l walltime=00:15:00
#PBS -P 50080337
#PBS -j oe
#PBS -o out-run-pred.txt
#PBS -q normal


module load miniforge3/23.10
conda activate aircraft_ai
cd scratch/
python pred_defects.py
