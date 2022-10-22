import torch.nn as nn
import torch.optim as optim
import time
import argparse
import DataPre
import torch
import numpy as np
from model import *
from torch.autograd import Variable
import csv
from math import pi

import os 

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adjust_learning_rate(optimizer, epoch,args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 40))
    lr = args.lr * ((1-(epoch /2001))**(0.9))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def getPSNR(t,v):
    m = np.max(t)-np.min(t)
    mse = np.mean((t-v)**2) 
    #print('mse', mse)
    psnr = 20*np.log10(m)-10*np.log10(mse)
    return psnr

def getAAD(t,v):
    t = torch.FloatTensor(t)
    #print('t',t)
    v = torch.FloatTensor(v)
    #print('v',v)
    cos = torch.sum(t*v,dim=0) / (torch.norm(t, dim=0) * torch.norm(v, dim=0) + 1e-10)
    #print('cos',cos)
    cos[cos>1] = 1
    cos[cos<-1] = -1
    aad = torch.mean(torch.acos(cos)).item() / pi
    return aad


def getRMSE(t,v):
    #print('gt', t)
    #print('predicts', v)
    error = np.sum(np.abs(t-v))/np.sum(np.abs(t))
    #print('error', error)
    return np.sqrt(np.mean(error))
######
######
def first_order_derivative(tensor,dim):
    '''
    This includes the function that computes the first-order derivative given a tensor 
    and its required axis (i.e., along x, y, or z axis). 
    You need to call this function nine times to get the Jacobian. Namely, 
    first-order derivative (u, dim=1) (du/dx), 
    first-order derivative (u, dim=2) (du/dy), 
    first-order derivative (u, dim=1) (du/dz),
    first-order derivative (v, dim=1) (dv/dx), 
    first-order derivative (v, dim=2) (dv/dy), 
    first-order derivative (v, dim=1) (dv/dz),  
    first-order derivative (w, dim=1) (du/dx), 
    first-order derivative (w, dim=2) (dw/dy), 
    first-order derivative (w, dim=1) (dw/dz).
    '''
    #print('input tensor shape', tensor.shape) # torch.Size([1, 1, 256, 224, 32])
    B,C,H,L,W = tensor.size()[0], tensor.size()[1], tensor.size()[2], tensor.size()[3],tensor.size()[4]
    diff = torch.full([B,C,H,L,W],2,dtype=torch.int)
    if dim == 2:
        tensor_h = torch.cat((tensor[:,:,1:H,:,:,],tensor[:,:,H-1:H,:,:,]),dim=2)
        tensor_h_ = torch.cat((tensor[:,:,0:1,:,:,],tensor[:,:,0:H-1,:,:,]),dim=2)
        diff[:,:,0:1,:,:,] = 1
        diff[:,:,H-1:H,:,:,] = 1
    elif dim == 3:
        tensor_h = torch.cat((tensor[:,:,:,1:L,:,],tensor[:,:,:,L-1:L,:,]),dim=3)
        tensor_h_ = torch.cat((tensor[:,:,:,0:1,:,],tensor[:,:,:,0:L-1,:,]),dim=3)
        diff[:,:,:,0:1,:,] = 1
        diff[:,:,:,L-1:L,:,] = 1
    elif dim == 4:
        tensor_h = torch.cat((tensor[:,:,:,:,1:W],tensor[:,:,:,:,W-1:W]),dim=4)
        tensor_h_ = torch.cat((tensor[:,:,:,:,0:1],tensor[:,:,:,:,0:W-1]),dim=4)
        diff[:,:,:,:,0:1] = 1
        diff[:,:,:,:,W-1:W] = 1
    else:
        assert "Not implemented!"

    #print('tensor_h shape', tensor_h.shape) #torch.Size([1, 1, 256, 224, 32])
    #print('tensor_h_ shape', tensor_h_.shape) #torch.Size([1, 1, 256, 224, 32])
    derivative = tensor_h - tensor_h_
    #print('derivative',derivative) 
    #print('derivative shape',derivative.shape) # torch.Size([1, 1, 128, 128, 128])
    #print(derivative.dtype)
    #print(diff.dtype)
    #print(diff.to(torch.float).dtype)
    #print('result', derivative/diff.to(torch.float).cuda())
    #exit()
    #print('diff.to(torch.float) shape', diff.to(torch.float).shape) # torch.Size([1, 1, 256, 224, 32])
    return derivative/diff.to(torch.float).cuda()
######

def Jacobian(tensor):
    #print('tensor shape', tensor.shape) #torch.Size([1, 3, 256, 224, 32])
    u11 = first_order_derivative(tensor[:,0:1,:,:,:],2)
    #print('u11 shape',u11.shape)
    u12 = first_order_derivative(tensor[:,0:1,:,:,:],3)
    #print('u12 shape',u12.shape)
    u13 = first_order_derivative(tensor[:,0:1,:,:,:],4)
    #print('u13 shape',u13.shape)

    v11 = first_order_derivative(tensor[:,1:2,:,:,:],2)
    #print('v11 shape',v11.shape)
    v12 = first_order_derivative(tensor[:,1:2,:,:,:],3)
    #print('v12 shape',v12.shape)
    v13 = first_order_derivative(tensor[:,1:2,:,:,:],4)
    #print('v13 shape',v13.shape)

    w11 = first_order_derivative(tensor[:,2:3,:,:,:],2)
    #print('w11 shape',w11.shape)
    w12 = first_order_derivative(tensor[:,2:3,:,:,:],3)
    #print('w12 shape',w12.shape)
    w13 = first_order_derivative(tensor[:,2:3,:,:,:],4)
    #print('w13 shape',w13.shape)
    #torch.cat((tensor[:,:,:,0:1],tensor[:,:,:,0:W-1]),dim=3)
    jacob = torch.cat((u11,u12,u13,v11,v12,v13,w11,w12,w13),dim=1) #torch.Size([1, 9, 128, 128, 128])
    #print('jacob shape',jacob.shape)
    #print('jacob ',jacob)
    return jacob


def train(model,args,dataset):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999))
    #criterion = nn.MSELoss()
    MSE_loss = nn.MSELoss()
    COS_loss = nn.CosineEmbeddingLoss()
    
    
    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([200,1,2000])
    #print(weights)
    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
    # weights[~mask] = 0
    weights = weights / weights.sum()

    #print(weights)
    for itera in range(1,args.epochs+1):
        print('======='+str(itera)+'========')
        train_loader = dataset.GetTrainingData()
        loss_mse = 0
        x = time.time()
        for batch_idx, (s,v) in enumerate(train_loader):
            ones = torch.ones((v.shape[0] * v.shape[2] * v.shape[3] * v.shape[4]))
            
            if args.cuda:
                s = s.cuda()
                v = v.cuda()
                ones = ones.cuda()
            #print('ones shape',ones.shape) #torch.Size([1, 3, 128, 128, 128])
            optimizer.zero_grad()
            v_ =  model(s)
            #print('v',v)
            #print('v shape',v.shape) #torch.Size([1, 3, 128, 128, 128])
            #print('v_',v_)
            #print('v_ shape',v_.shape) #torch.Size([1, 3, 128, 128, 128])
            
            magnitude_loss = MSE_loss(v_,v)
            
            #print('magnitude_loss',magnitude_loss)

            jacob_pred = Jacobian(v_)
            #print('jacob_pred',jacob_pred)
            #print('jacob_pred shape',jacob_pred.shape)
            jacob_gt = Jacobian(v)
            #print('jacob_gt',jacob_gt)
            #print('jacob_gt shape',jacob_gt.shape)
            Jacob_loss =  MSE_loss(jacob_pred,jacob_gt)
            #print('Jacob_loss',Jacob_loss)
            #exit()
            
            v_pred = v_.reshape(v_.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
            v_gt = v.reshape(v.shape[0], 3, -1).transpose(1, 2).reshape(-1, 3)
            cos_loss = COS_loss(v_pred, v_gt, ones)
            #print('cos_loss',cos_loss)

            
            loss_all = weights[0]*magnitude_loss + weights[1]*cos_loss + weights[2]*Jacob_loss
            loss_all.backward()
            loss_mse += loss_all.mean().item()
            optimizer.step()
        y = time.time()
        print("Time = "+str(y-x))
        print("Loss = "+str(loss_mse))
        with open("Loss_v100.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=["Epochs", "Loss", "Time"])
                writer.writeheader()
                writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerow([itera, loss_mse, y-x])
        adjust_learning_rate(optimizer,itera,args)
        if (itera>999 and itera%100==0) or itera==10:
            torch.save(model.state_dict(),args.model_path+args.dataset+'/supercurrent-'+str(itera)+'.pth')


def inference(model,Vec,args,epoch):
    for i in range(0,len(Vec.scalar)):
        print(i)
        x = time.time()
        s = np.zeros((1,1,Vec.dim[0],Vec.dim[1],Vec.dim[2]))
        s[0] = Vec.scalar[i]
        s = torch.FloatTensor(s).cuda()
        #vec = concatsubvolume(model,s,[128,128,128],args)
        ######
        vec = model(s)
        vec = vec.cpu().detach().numpy()
        ######
        y = time.time()
        print('Time: ', str(y-x))  
        data = np.asarray(vec,dtype='<f')
        data = data.flatten('F')
        data.tofile(args.result_path+'/'+args.dataset+'{:03d}'.format((i+1)*9)+'_kv2v_newloss.vec',format='<f')
        

        ######
        gt_path = '/afs/crc.nd.edu/group/vis/Vector/V/supercurrent/' + '{:03d}'.format((i+1)*9)+ '.vec'
        gt = np.fromfile(gt_path,dtype='<f')

        result_path = args.result_path+'/'+args.dataset+'{:03d}'.format((i+1)*9)+'_kv2v_newloss.vec'
        result = np.fromfile(result_path,dtype='<f')

        res_PSNR = getPSNR(gt,result)
        print('Inference PSNR result')
        print('res_PSNR is: ', res_PSNR)
        res_AAD = getAAD(gt,result)
        print('Inference AAD result')
        print('res_AAD is: ', res_AAD)
        res_RAE = getRMSE(gt,result)
        print('Inference RAE result')
        print('res_RAE is: ', res_RAE)

        File_name = str(epoch) + "_Inference_PSNR_result.csv"
        with open(File_name, "a") as f:
            writer = csv.DictWriter(f, fieldnames=["test_file","Result_PSNR", "Time", "Result_AAD", "res_RAE"])
            writer.writeheader()
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow([i+1,res_PSNR,  y-x, res_AAD, res_RAE])

def concatsubvolume(model,data,win_size,args):
    x,y,z = data.size()[2],data.size()[3],data.size()[4]
    w = np.zeros((win_size[0],win_size[1],win_size[2]))
    for i in range(win_size[0]):
        for j in range(win_size[1]):
            for k in range(win_size[2]):
                dx = min(i,win_size[0]-1-i)
                dy = min(j,win_size[1]-1-j)
                dz = min(k,win_size[2]-1-k)
                d = min(min(dx,dy),dz)+1
                w[i,j,k] = d
    w = w/np.max(w)
    avI = np.zeros((x,y,z))
    pmap= np.zeros((1,3,x,y,z))
    avk = 4
    for i in range((avk*x-win_size[0])//win_size[0]+1):
        for j in range((avk*y-win_size[1])//win_size[1]+1):
            for k in range((avk*z-win_size[2])//win_size[2]+1):
                si = (i*win_size[0]//avk)
                ei = si+win_size[0]
                sj = (j*win_size[1]//avk)
                ej = sj+win_size[1]
                sk = (k*win_size[2]//avk)
                ek = sk+win_size[2]
                if ei>x:
                    ei= x
                    si=ei-win_size[0]
                if ej>y:
                    ej = y
                    sj = ej-win_size[1]
                if ek>z:
                    ek = z
                    sk = ek-win_size[2]
                d0 = data[:,:,si:ei,sj:ej,sk:ek]
                with torch.no_grad():
                    v = model(d0)
                k = np.multiply(v.cpu().detach().numpy(),w)
                avI[si:ei,sj:ej,sk:ek] += w
                pmap[:,:,si:ei,sj:ej,sk:ek] += k
    result = np.divide(pmap,avI)
    return result

