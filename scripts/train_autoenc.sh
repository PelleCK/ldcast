#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=test_LDCast_VAE_KNMIdata
#SBATCH --output=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/results/logs/%x.out
#SBATCH --error=/vol/knmimo-nobackup/users/pkools/thesis-forecasting/ldcast/results/logs/%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pelle.kools@ru.nl

MODEL_DIR="../results/model_checkpoints/${SLURM_JOB_NAME}/"

# # check if progress checkpoint has been stored
# # if so, use it as initial checkpoint
# if [ -f "${MODEL_DIR}/latest.ckpt" ]; then
#     CKPT_ARG="--ckpt_path ${MODEL_DIR}/latest.ckpt"
# else
#     CKPT_ARG=""
# fi

source ../.ldcast_venv/bin/activate
python train_autoenc.py --train_IDs_fn='../data/knmi_data/data_splits/train2015_2018_3y_30m.npy' --val_IDs_fn='../data/knmi_data/data_splits/val2019_3y_30m.npy' --batch_size=32 --model_dir=$MODEL_DIR
