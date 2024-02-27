#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/myjob-%j.out
#SBATCH --error=./logs/myjob-%j.err

