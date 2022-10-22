#!/bin/csh
#$ -M pgu@nd.edu
#$ -q gpu@qa-v100-002 -l gpu=1
#$ -m abe
#$ -r y
#$ -N supercurrent_kv2v_newloss_true_200_1_2000_normalize

module load pytorch/1.1.0	         # Required modules
python3 main.py
#python3 gt_to_scalar_gt.py
#tensorflow/1.3
