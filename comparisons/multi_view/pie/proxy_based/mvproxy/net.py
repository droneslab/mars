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
from alive_progress import alive_bar
from util import Resnext_avg_mvproxy, compute_acc, mvDataset, load_data


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
parser.add_argument('--trial', action='store', default=1, type=int, help='weight (default: 1.0)')
arg = parser.parse_args()


def main():
    # create model directory to store/load old model
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')

	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    ch = logging.FileHandler('log/logfile_'+arg.net+'_'+arg.dataset + '_' + str(arg.trial)  + '.log')
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
    torch.cuda.set_device(0)

    # dataset directory
    # if arg.dataset == 'ModelNet':
    #     pickle_filename = '../../modelnet.pickle' 
    #     output_class = 40
    #     input_view = 12

    # pickle_filename = '../../ObjectPI.pickle'
    # output_class = 25
    # input_view = 8
    
    pickle_filename = '../../lunar_hirise.pickle'
    output_class = 4551
    input_view = 7
    
    train, test = load_data(pickle_filename)
    
    # import numpy as np
    # print(train['objs'])
    # print(np.array(train['objs']).shape)
    # print(np.array(train['labels']).shape)
    # quit()
    
    # if arg.train_f:
    dataset_train = mvDataset(train, max_view=input_view, transform =data_transforms['train'])
    dataloader_trn = DataLoader(dataset_train, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    # dataset_val = mvDataset(test, max_view=input_view,  transform =data_transforms['val'])
    # dataloader_val = DataLoader(dataset_val, batch_size=arg.batchSize, shuffle=True, num_workers=4)
        
    # dataset_test = mvDataset(test, max_view=input_view,  transform =data_transforms['test'])
    # dataloader_test = DataLoader(dataset_test, batch_size=arg.batchSize, shuffle=False, num_workers=4)
    
    # input_view = 7
    # dataset_train = mvDataset2('train', max_view=input_view, transform =data_transforms['train'])
    # dataloader_trn = DataLoader(dataset_train, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    # dataset_val = mvDataset2('val', max_view=input_view, transform =data_transforms['train'])
    # dataloader_val = DataLoader(dataset_train, batch_size=arg.batchSize, shuffle=True, num_workers=4)
    # output_class = dataset_train.num_craters
    
    logger.info("Output classes: {}".format(output_class))
    logger.info("Input view: {}".format(input_view))

    model = Resnext_avg_mvproxy(input_view, output_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    model_path = 'model/model_' +'resnext'+'_'+'lunar_hirise' + '_' + str(1) +'.pt'
    
    # training
    print("Start Training")
    logger.info("Start Training")

    epochs = arg.epochs if arg.train_f else 0
    min_accuracy = 0.0
    
    correct, ave_loss = 0.0, 0.0
    for epoch in range(epochs):
        # training
        batch_idx = 0
        overall_acc = 0
        model.train()
        
        with alive_bar(len(dataloader_trn)) as bar:

            for batch_idx, (batch_x, batch_y) in enumerate(dataloader_trn):
                optimizer.zero_grad()
                # for gpu mode
                batch_x.requires_grad = True
                input = batch_x

                # use cross entropy loss
                # criterion_nll = nn.NLLLoss()            
                shape_feature, class_feature = model(input)
                # y = batch_y.numpy()
                y = batch_y
    
                loss = None
                for b in range(y.shape[0]):
                    loss_shape_nominator, loss_shape_denominator = None, None
                    for o in range(output_class):
                                            
                        neg_dist = -torch.sum((shape_feature[b]-class_feature[o])**2)
                        exp_neg_dist = torch.clamp(torch.exp(neg_dist), min=1e-16)
                        if o == y[b]:
                            loss_shape_nominator = exp_neg_dist
                        else:
                            if loss_shape_denominator is None:
                                loss_shape_denominator = exp_neg_dist
                            else:
                                loss_shape_denominator += exp_neg_dist
                                
                    loss_shape = -(torch.log(loss_shape_nominator) - torch.log(loss_shape_denominator))
                    
                    if loss is None:
                        loss = loss_shape
                    else:
                        loss += loss_shape
                        
                loss /= y.shape[0]
                loss.backward()              
                optimizer.step()     
                
                
                if batch_idx%10==0:
                    # accuracy = compute_acc(shape_feature, class_feature, y)*1.0/y.shape[0]
                    accuracy = 0
                    print('epoch:{}, batch index: {}, loss:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy()))
                    logger.info('==>>> epoch:{}, batch index: {}, loss:{}, accuracy:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy() ,accuracy))
                batch_idx += 1
                bar()
            
           
            
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
        print('==>>>val accuracy:{}'.format(accuracy))
        logger.info('==>>>val accuracy:{}'.format(accuracy))

            
    # Testing
    # print("Start Testing for Best Model")
    # logger.info("Start Testing for Best Model")
    # if os.path.isfile(model_path):
    #     print("Loading model")
    #     logger.info("Loading model")
    #     model.load_state_dict(torch.load(model_path))
        
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