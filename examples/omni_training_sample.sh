#!/bin/bash
#SBATCH --job-name=v4
#SBATCH --output=my_job_v4.txt
#SBATCH --error=my_job_err_v4.txt
#SBATCH --nodes=1
#SbATCH --mem=64
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 
#SBATCH --mail-type=END
#SBATCH --mail-user=syed.kazim@uni-siegen.de

# module load miniconda3
# deactivate env
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda deactivate

# activate conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 2025_Optimizing_Phase_Mask

export QT_QPA_PLATFORM=offscreen
# python v9_synthetic_Flow_training.py --name 'RAFT_synthetic_beads_v4_hblur_RAFT' --loss 'l1' --ckpt_load 'no' --ckpt_save 'yes' --epochs 1500 --iter_pm 3 --lr 0.0001
python v10_synth_cell_flow_training.py --name 'RAFT_synthetic_cells_v11_blur_RAFT_lnoise_rcrop_rnfocus' --loss 'l1' --ckpt_load 'no' --ckpt_save 'yes' --epochs 1500 --iter_pm 3 --lr 0.0001