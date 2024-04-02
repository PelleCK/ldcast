#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/scripts/logs/myjob-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/scripts/logs/myjob-%j.err

source ../.ldcast_venv/bin/activate
python forecast_demo.py --num_diffusion_iters=10
