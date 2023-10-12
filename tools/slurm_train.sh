#!/bin/bash

#SBATCH --output="/cluster/work/cvl/qutang/slurm_logs/%j.out"
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=128G
#SBATCH --gpus=8
#SBATCH --gres=gpumem:24G
#SBATCH --job-name=mmseg

# data
module load pigz gcc/8.2.0 cuda/11.7.0 cudnn/8.8.1.3
#tar -I pigz -xvf /cluster/work/cvl/qutang/ADEChallengeData2016.tar.gz -C ${TMPDIR}/
#tar -I pigz -xvf /cluster/work/cvl/qutang/cityscapes.tar.gz -C ${TMPDIR}/
tar -I pigz -xvf /cluster/work/cvl/qutang/pcontext.tar.gz -C ${TMPDIR}/

# config
GPUS=8
PORT=29504

#CONFIG="configs/cft/cft_r50-d32_512x512_ade20k.py" # 11G, 35h
#CONFIG="configs/cft/cft_r101-d32_512x512_ade20k.py"  # 11G, 35h
#CONFIG="configs/cft/cft_mit-b2_512x512_ade20k.py"  # 11G, 35h
#CONFIG="configs/cft/cft_mit-b5_640x640_ade20k.py" # 24G, 65h
#CONFIG="configs/cft/ocrnet_r50-d8_512x512_ade20k.py"  # 24G, 15h
#CONFIG="configs/cft/cft_swin-large-4-12_640x640_ade20k.py" # 24G, 65h
#CONFIG="configs/cft/cft_swin-large-4-12_1024x1024_citys.py" # 24G, 55h
#CONFIG="configs/cft/cft_swin-large-4-12_480x480_pascal-context.py" # 24G, 24h
CONFIG="configs/cft/cft_swin-large-4-12_480x480_pascal-context-59.py" # 24G, 24h

# log directory
timeStamp=$(date +%Y%m%d_%H%M%S)
mkdir ${TMPDIR}/work_dir_$timeStamp

# env & run
source /cluster/home/qutang/torch/bin/activate
cd /cluster/home/qutang/mmsegmentation || exit
PORT=$PORT bash tools/dist_train.sh $CONFIG $GPUS --work-dir ${TMPDIR}/work_dir_$timeStamp

# copy logs back
rsync -aPq ${TMPDIR}/work_dir_$timeStamp /cluster/work/cvl/qutang/mmseg_logs/