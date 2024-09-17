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
import time

from util import Resnext_avg_mvtc, compute_acc, mvDataset, load_data


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 40)')
parser.add_argument('--batchSize', action='store', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.00001, type=float, help='learning rate (default: 0.00001)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--w', '--weight-decay', action='store', default=0, type=float, help='regularization weight decay (default: 0.0)')
parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--preTrained_f', action='store_false', default=True, help='Flag to pretrained model (default: True)')
# parser.add_argument('--gpu_num','--list', type=int , nargs='+', help='gpu_num', required=True)
parser.add_argument("--net", default='VGG_avg', const='VGG_avg',nargs='?', choices=['VGG_avg'], help="net model(default:VGG_avg)")
parser.add_argument("--dataset", default='ObjectPI', const='ObjectPI',nargs='?', choices=['ModelNet', 'ObjectPI'], help="Dataset (default:ObjectPI)")
parser.add_argument('--margin', action='store', default=1, type=float, help='margin (default: 1.0)')
parser.add_argument('--trial', action='store', default=1, type=int, help='weight (default: 1.0)')
arg = parser.parse_args()

# logdir = '/user/tbchase/projects/data/landmark_matching/pie_piproxy_resnext/'
logdir = '../../../../../data/training_logs/latest/pie/'

def main():
    # create model directory to store/load old model
    if not os.path.exists(f'{logdir}/model'):
        os.makedirs(f'{logdir}/model')
    if not os.path.exists(f'{logdir}/log'):
        os.makedirs(f'{logdir}/log')

	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_'+arg.net+'_'+arg.dataset + '_' + str(arg.margin) + '_' + str(arg.trial)  + '.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Momentum: {}".format(arg.m))
    logger.info("Regularization Weight Decay: {}".format(arg.w))
    logger.info("Classifier: "+arg.net)
    logger.info("Dataset: "+arg.dataset)
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    
    # Batch size setting
    batch_size = arg.batchSize
    
    # load the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # if len(arg.gpu_num) == 1:
    #     torch.cuda.set_device(arg.gpu_num[0])

    # # dataset directory
    # if arg.dataset == 'ModelNet':
    #     pickle_filename = '../../modelnet.pickle' 
    #     output_class = 40
    #     input_view = 12

    # elif arg.dataset == 'ObjectPI':
    #     pickle_filename = '../../ObjectPI.pickle'
    #     output_class = 25
    #     input_view = 8
    
    pickle_filename = '../../lunar_hirise.pickle'
    output_class = 4551
    input_view = 7
        

    train, test = load_data(pickle_filename)      
    # if arg.train_f:
    dataset_train = mvDataset(train, max_view=input_view, transform =data_transforms['train'])
    dataloader_trn = DataLoader(dataset_train, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    # dataset_val = mvDataset(test, max_view=input_view,  transform =data_transforms['val'])
    # dataloader_val = DataLoader(dataset_val, batch_size=arg.batchSize, shuffle=True, num_workers=4)
        
    # dataset_test = mvDataset(test, max_view=input_view,  transform =data_transforms['test'])
    # dataloader_test = DataLoader(dataset_test, batch_size=arg.batchSize, shuffle=False, num_workers=4)    
        
    logger.info("Output classes: {}".format(output_class))
    logger.info("Input view: {}".format(input_view))

    # if arg.net == 'VGG_avg':
        # model = VGG_avg_piproxy(input_view, output_class)
    #optimizer = torch.optim.Adam(params, lr=arg.lr)
    model = Resnext_avg_mvtc(input_view, output_class).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.w)
    #optimizer = optim.SGD(model.parameters(), lr=arg.lr, momentum=arg.m)
    # for gpu mode
    # if arg.useGPU_f:
    #     if len(arg.gpu_num) > 1:
    #         model = nn.DataParallel(model, arg.gpu_num)
    #     model.cuda()
    # # for cpu mode
    # else:
    #     model
    # model_path = 'model/model_'+arg.net+'_'+arg.dataset + '_' + str(arg.alpha)+ '_' + str(arg.beta) + '_' + str(arg.trial)  +'.pt'
    model_path = f'{logdir}/model_' +'resnext'+'_'+'lunar_hirise' + '_' + str(1) +'.pt'

    # training
    print("Start Training")
    logger.info("Start Training")
        
        
    epochs = arg.epochs if arg.train_f else 0
    min_accuracy = 0.0
    

    correct, ave_loss = 0.0, 0.0
    for epoch in range(epochs):
        s = time.time()
        # training
        batch_idx = 0
        overall_acc = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader_trn):
            optimizer.zero_grad()
            # for gpu mode
            batch_x.requires_grad = True
            input = batch_x.cuda()

            # use cross entropy loss
            # criterion_nll = nn.NLLLoss()
            
            shape_feature, class_feature = model(input)
            # y = batch_y.numpy()
            y = batch_y.cuda()
            
            loss = None
            for b in range(y.shape[0]):
                min_dis = None
                for o in range(output_class):
                    if not o == y[b]:
                        dist = torch.sum((shape_feature[b]-class_feature[o])**2)
                        if min_dis is None:
                            min_dis = dist
                        else:
                            min_dis = torch.min(min_dis, dist)
                          
                dist_y = torch.sum((shape_feature[b]-class_feature[y[b]])**2)
                
                if loss is None:
                    loss = torch.clamp(dist_y -min_dis + arg.margin, min=0)

                else:
                    loss += torch.clamp(dist_y -min_dis + arg.margin, min=0)
                
            loss /= y.shape[0]
            loss.backward()              
            optimizer.step()     
            
            
            
            if batch_idx%10==0:
                # accuracy = 1.0*compute_acc(image_feature, class_feature, y)*1.0/y.shape[0]
                accuracy = 0
                print('==>>> epoch:{}, batch index: {}/{}, loss:{}, epoch elapsed:{}'.format(epoch,batch_idx, len(dataloader_trn), loss.cpu().detach().numpy(), time.time()-s))
                logger.info('==>>> epoch:{}, batch index: {}/{}, loss:{}, epoch elapsed:{}'.format(epoch,batch_idx, len(dataloader_trn), loss.cpu().detach().numpy(), time.time()-s))
            batch_idx += 1
        print(f'----- Epoch time: {time.time()-s}')
            
        # Validation (always save the best model)
        # print("Start Validation")
        # logger.info("Start Validation")
        
        #     # switch to evaluate mode
        # model.eval()
        # correct = 0.0

        # for batch_idx, (batch_x, batch_y) in enumerate(dataloader_val):
        #     with torch.no_grad():
        #         input = batch_x.cuda()
        #         shape_feature, class_feature = model(input)
        #         y = batch_y.numpy()
        #         correct += compute_acc(shape_feature, class_feature, y)
                    
        # accuracy = 1.0*correct*1.0/len(dataset_val)

        # if accuracy >= min_accuracy:
        #     min_accuracy = accuracy
        #     # save the model if it is better than current one
        torch.save(model.state_dict(), model_path)
        # print('==>>>test accuracy:{}'.format(accuracy))
        # logger.info('==>>>test accuracy:{}'.format(accuracy))

            
    # Testing
    # print("Start Testing for Best Model")
    # logger.info("Start Testing for Best Model")
    # if os.path.isfile(model_path):
    #     print("Loading model")
    #     logger.info("Loading model")
    #     model.load_state_dict(torch.load(model_path))
    # # switch to evaluate mode
    # model.eval()
    # correct = 0.0

    # for batch_idx, (batch_x, batch_y) in enumerate(dataloader_test):
    #     with torch.no_grad():
    #         input = batch_x.cuda()
    #         shape_feature, class_feature = model(input)
    #         y = batch_y.numpy()
    #         correct += compute_acc(shape_feature, class_feature, y)
        
    # accuracy = 1.0*correct*1.0/len(dataset_test)

    # print('==>>>test accuracy:{}'.format(accuracy))
    # logger.info('==>>>test accuracy:{}'.format(accuracy))

        
if __name__ == "__main__":
    main()