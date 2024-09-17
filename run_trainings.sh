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

WHERE=$1
if [ "$WHERE" = "vm2" ]; then
    source ~/.venv/bin/activate
    CFG=../cfg/vm2.yaml
elif [ "$WHERE" = "laptop" ]; then
    source /home/tj/envs/py3.10/bin/activate
    CFG=../cfg/laptop.yaml
elif [ "$WHERE" = "ccr" ]; then
    CFG=../cfg/ccr.yaml
elif [ "$WHERE" = "desktop" ]; then
    source /home/tj/sto_seg/env/bin/activate
    CFG=../cfg/desktop.yaml
fi

EPOCHS=150
# for DATASET in hirise lunar; do
# for DATASET in lunar; do
#     if [ $DATASET = hirise ] || [ $DATASET = aid ] || [ $DATASET = resisc45 ]; then
#         BSIZE=32
#     else
#         BSIZE=128
#     fi

#     for ML in proxyanchor proxynca++ supcon subcenterarcface circle drms synproxy pnp; do
#         python train_and_eval.py --cfg $CFG --epochs $EPOCHS --batch_size $BSIZE --conv_type conv --att_type se --ml_loss $ML --dataset $DATASET --save
#         python train_and_eval.py --cfg $CFG --epochs $EPOCHS --batch_size $BSIZE --conv_type ric  --att_type ca --ml_loss $ML --dataset $DATASET --save
#     done
# done

for DATASET in hirise lunar; do
    if [ $DATASET = hirise ] || [ $DATASET = aid ] || [ $DATASET = resisc45 ]; then
        BSIZE=32
    else
        BSIZE=128
    fi

    for ML in ntxent proxynca++ supcon subcenterarcface circle drms synproxy pnp; do
        python train_and_eval.py --cfg $CFG --epochs $EPOCHS --batch_size $BSIZE --conv_type ric  --att_type ca --ml_loss $ML --dataset $DATASET --save \
                                    --aar --ch_gamma 0.15 --sp_gamma 0.15
    done
done
