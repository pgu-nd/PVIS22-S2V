from model import *
from train import *
import os
import argparse
from DataPre import DataSet
import torch

parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "TSR-VFD"')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate of TSR-VFD')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=2001, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--dataset', type=str, default='supercurrent', metavar='N',
                    help='the data set we used for training TSR-VFD')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='the number of crops for a pair of data')
parser.add_argument('--init_channels', type=int, default=16, metavar='N',
                    help='the number of crops for a pair of data')
parser.add_argument('--samples', type=int, default=40, metavar='N',
                    help='the samples we used for training TSR-VFD')
parser.add_argument('--data_path', type=str, default='/afs/crc.nd.edu/user/j/jhan5/vis/', metavar='N',
                    help='the path where we stored the saved model')
parser.add_argument('--model_path', type=str, default='/afs/crc.nd.edu/user/p/pgu/Research/Scalar2Vector/Supercurrent_kV2V_2000_newloss_true_200_1_2000_normalize/saved_model/', metavar='N',
                    help='the path where we stored the saved model')
parser.add_argument('--result_path', type=str, default='/afs/crc.nd.edu/user/p/pgu/Research/Scalar2Vector/Supercurrent_kV2V_2000_newloss_true_200_1_2000_normalize/result/', metavar='N',
                    help='the path where we stored the synthesized data')
parser.add_argument('--train', type=str, default='train', metavar='N',
                    help='')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def main():
    if not os.path.exists(args.model_path+args.dataset):
        os.mkdir(args.model_path+args.dataset)
    model = kV2V(1,3,64)
    model.apply(weights_init_normal)
    model.cuda()
    VectorData = DataSet(args)
    VectorData.ReadData()
    
    train(model,args,VectorData)  
    
    

def GetResult():
    for epoch in range(2000, 2100, 100):
        print('epoch', epoch)
        model = kV2V(1,3,64)
        model_dict = torch.load(args.model_path+args.dataset+'/supercurrent-'+str(epoch)+'.pth',map_location=lambda storage, loc:storage)
        model.load_state_dict(model_dict)
        model.cuda()
        VectorData = DataSet(args)
        VectorData.ReadData()
        inference(model,VectorData,args,epoch)
        #GetMetrics(VectorData,args)

if __name__== "__main__":
    if args.train=='train':
        main()
    else:
        GetResult()




        