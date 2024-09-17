# To check the result from the pretrained model in log
python net.py --dataset 'ModelNet' --trial 1 -e 40 --train_f --gpu_num 0
python net.py --dataset 'ObjectPI' --trial 1 -e 40 --train_f --gpu_num 0

#To check robustness to the number of views in log_robustnesss
python net_robustness.py --dataset 'ModelNet' --trial 1 --gpu_num 0
python net_robustness.py --dataset 'ObjectPI' --trial 1 --gpu_num 0

# To train your own model
# python net.py --dataset 'ModelNet'  --trial 1 -e 40 --gpu_num 0
# python net.py --dataset 'ObjectPI'  --trial 1 -e 40 --gpu_num 0