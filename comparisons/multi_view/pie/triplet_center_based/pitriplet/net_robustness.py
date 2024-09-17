from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torchvision.models as models
import argparse
import logging
import torchvision.transforms as transforms
import os
import torch.utils.data
import torch.cuda as cuda
import pickle
from math import exp
from torch.utils.data import Dataset, DataLoader

from util_robustness import VGG_avg_pitc, compute_acc, mvDataset, load_data

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.00001, type=float, help='learning rate (default: 0.00001)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--w', '--weight-decay', action='store', default=0, type=float, help='regularization weight decay (default: 0.0)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--preTrained_f', action='store_false', default=True, help='Flag to pretrained model (default: True)')
parser.add_argument('--gpu_num','--list', type=int , nargs='+', help='gpu_num', required=True)
parser.add_argument("--net", default='VGG_avg', const='VGG_avg',nargs='?', choices=['VGG_avg'], help="net model(default:VGG_avg)")
parser.add_argument("--dataset", default='ObjectPI', const='ObjectPI',nargs='?', choices=['ModelNet', 'miro', 'ObjectPI'], help="Dataset (default:ObjectPI)")
parser.add_argument('--margin', action='store', default=1, type=float, help='margin (default: 1.0)')
parser.add_argument('--alpha', action='store', default=1, type=float, help='alpha (default: 1.0)')
parser.add_argument('--beta', action='store', default=1, type=float, help='beta (default: 1.0)')
parser.add_argument('--trial', action='store', default=1, type=int, help='weight (default: 1.0)')
arg = parser.parse_args()


def main():
    # create model directory to store/load old model
    if not os.path.exists('log_robustness'):
        os.makedirs('log_robustness')

	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log_robustness/logfile_'+arg.net+'_'+arg.dataset + '_' + str(arg.margin)+ '_' + str(arg.alpha)+ '_' + str(arg.beta) + '_' + str(arg.trial)  + '.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    
    # Batch size setting
    batch_size = arg.batchSize
    
    # load the data
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    if len(arg.gpu_num) == 1:
        torch.cuda.set_device(arg.gpu_num[0])

    # dataset directory
    if arg.dataset == 'ModelNet':
        pickle_filename = '../../modelnet.pickle' 
        output_class = 40
        input_view = 12

    elif arg.dataset == 'ObjectPI':
        pickle_filename = '../../ObjectPI.pickle'
        output_class = 25
        input_view = 8


    train, test = load_data(pickle_filename)      
        
    dataset_test = mvDataset(test, max_view=input_view,  transform =data_transforms['test'])
    dataloader_test = DataLoader(dataset_test, batch_size=arg.batchSize, shuffle=False, num_workers=4)
        

    if arg.net == 'VGG_avg':
        model = VGG_avg_pitc(input_view, output_class)

    # for gpu mode
    if arg.useGPU_f:
        if len(arg.gpu_num) > 1:
            model = nn.DataParallel(model, arg.gpu_num)
        model.cuda()
    # for cpu mode
    else:
        model
    model_path = 'model/model_'+arg.net+'_'+arg.dataset + '_' + str(arg.margin)+ '_' + str(arg.alpha)+ '_' + str(arg.beta) + '_' + str(arg.trial)  +'.pt'
      
    # Testing
    print("Start Testing for Best Model")
    if os.path.isfile(model_path):
        print("Loading model")
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    else:
        print("No model")
        return

    model.eval()
    
    for sample_view in range(1,input_view+1):        
        logger.info("Sample View: {}".format(sample_view))
        correct = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader_test):
            with torch.no_grad():
                if sample_view == 1:
                    input = batch_x.cuda()
                    image_feature, class_feature = model(input,sample_view)
                    batch_y = batch_y.view(batch_y.shape[0],1).repeat(1,input_view).view(batch_y.shape[0]*input_view)
                    y = batch_y.numpy()
                    correct += compute_acc(image_feature, class_feature, y)
                else:
                    input = batch_x.cuda()
                    shape_feature, class_feature = model(input,sample_view)
                    y = batch_y.numpy()
                    correct += compute_acc(shape_feature, class_feature, y)
                    
        if sample_view == 1:
            accuracy = 1.0*correct*1.0/len(dataset_test)/float(input_view)
        else:
            accuracy = 1.0*correct*1.0/len(dataset_test)
            
        print('==>>> accuracy:{:0.4f}'.format(accuracy))
        logger.info('==>>> accuracy:{:0.4f}'.format(accuracy))

        
if __name__ == "__main__":
    main()