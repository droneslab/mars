#!/bin/bash
#SBATCH --partition=hal --qos=hal
#SBATCH --cluster=faculty
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --job-name="landmark_recognition"
#SBATCH --output=out.txt
#SBATCH --mail-user=tbchase@buffalo.edu
#SBATCH --mail-type=ALL

cd src/


# python train_and_eval.py --cfg ../cfg/vm2.yaml --epochs 150 --batch_size 32 --conv_type conv --att_type se \
#         --ml_loss pairloss --dataset aid

python train_and_eval.py --cfg ../cfg/vm2.yaml --epochs 150 --batch_size 32 --conv_type conv --att_type se \
        --ml_loss ride --dataset aid

python train_and_eval.py --cfg ../cfg/vm2.yaml --epochs 150 --batch_size 32 --conv_type conv --att_type se \
        --ml_loss pairloss --dataset resisc45

python train_and_eval.py --cfg ../cfg/vm2.yaml --epochs 150 --batch_size 32 --conv_type conv --att_type se \
        --ml_loss ride --dataset resisc45
