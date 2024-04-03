#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/scripts/logs/myjob-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/scripts/logs/myjob-%j.err

source ../.ldcast_venv/bin/activate
python forecast_demo.py --t0="2019/10/28 11:25" --draw_border=False, --data_dir="../data/demo/knmi/" --num_diffusion_iters=10 --dataset_type="knmi"
