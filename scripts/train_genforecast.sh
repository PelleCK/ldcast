#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/scripts/logs/train_genforecast-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/scripts/logs/train_genforecast-%j.err

source ../.ldcast_venv/bin/activate
python train_genforecast.py --config="../config/genforecast-radaronly-128x128-20step.yaml" --batch_size=16 --future_timesteps=5
