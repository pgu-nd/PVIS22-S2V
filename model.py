import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math
import torch.nn.utils.spectral_norm as spectral_norm

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        m.weight.data.normal_(0,0.01)
    elif classname.find("Linear")!=-1:
        m.weight.data.normal_(0,0.01)
    elif classname.find("BatchNorm")!=-1:
        m.weight.data.normal_(1.0,0.01)
        #m.bias.data.constant_(0.0)

def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_normal(m.weight.data,a=0,mode="fan_in")
    elif classname.find("Linear")!=-1:
        init.kaiming_normal(m.weight.data,a=0,mode="fan_in")
    elif classname.find("BatchNorm")!=-1:
        init.normal(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)


def BuildResidualBlock(channels,dropout,kernel,depth,bias):
  layers = []
  for i in range(int(depth)):
    layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
               #nn.BatchNorm3d(channels),
               nn.ReLU(True)]
    if dropout:
      layers += [nn.Dropout(0.5)]
  layers += [nn.Conv3d(channels,channels,kernel_size=kernel,stride=1,padding=kernel//2,bias=bias),
             #nn.BatchNorm3d(channels),
           ]
  return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
  def __init__(self,channels,dropout,kernel,depth,bias):
    super(ResidualBlock,self).__init__()
    self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

  def forward(self,x):
    out = x+self.block(x)
    return out

class Block(nn.Module):
    def __init__(self,inchannels,outchannels,dropout,kernel,bias,depth,mode,factor=2):
        super(Block,self).__init__()
        layers = []
        for i in range(int(depth)):
            layers += [
                       spectral_norm(nn.Conv3d(inchannels,inchannels,kernel_size=kernel,padding=kernel//2,bias=bias),eps=1e-4),
                       nn.InstanceNorm3d(inchannels),
                       nn.LeakyReLU(0.2,inplace=True)]
            if dropout:
                layers += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*layers)
        if factor == 2:
            self.stride = 2
            self.padding = 1
        elif factor == 4:
            self.stride = 1
            self.padding = 0
        if mode == 'down':
            self.conv1 = spectral_norm(nn.Conv3d(inchannels,outchannels,4,stride=self.stride,padding=self.padding,bias=bias),eps=1e-4)
            self.conv2 = spectral_norm(nn.Conv3d(inchannels,outchannels,4,stride=self.stride,padding=self.padding,bias=bias),eps=1e-4)
        elif mode == 'up':
            self.conv1 = spectral_norm(nn.ConvTranspose3d(inchannels,outchannels,4,stride=self.stride,padding=self.padding,bias=bias),eps=1e-4)
            self.conv2 = spectral_norm(nn.ConvTranspose3d(inchannels,outchannels,4,stride=self.stride,padding=self.padding,bias=bias),eps=1e-4)
        elif mode == 'same':
            self.conv1 = nn.Sequential(*[nn.ReplicationPad3d(kernel//2),
                                       spectral_norm(nn.Conv3d(inchannels,outchannels,kernel_size=kernel,bias=bias),eps=1e-4)
                                       ])

            self.conv2 = nn.Sequential(*[nn.ReplicationPad3d(kernel//2),
                                       spectral_norm(nn.Conv3d(inchannels,outchannels,kernel_size=kernel,bias=bias),eps=1e-4)
                                       ])

            self.conv3 = nn.Sequential(*[nn.ReplicationPad3d(kernel//2),
                                       spectral_norm(nn.Conv3d(2*outchannels,outchannels,kernel_size=kernel,bias=bias),eps=1e-4)
                                       ])

    def forward(self,x):
        y = self.model(x)
        y = self.conv1(y)
        x = self.conv2(x)
        #return self.conv3(torch.cat([x,y],dim=1))
        return x+y


class kV2V(nn.Module):
    def __init__(self,inc,outc,init_channels):
        super(kV2V,self).__init__()
        ### first model 
        self.input_pool = nn.Conv3d(inc,init_channels,4,2,1)

        self.conv1 = nn.Conv3d(init_channels,init_channels,4,2,1) 
        self.conv2 = nn.Conv3d(init_channels,2*init_channels,4,2,1) 
        self.conv3 = nn.Conv3d(2*init_channels,4*init_channels,4,2,1)
        self.conv4 = nn.Conv3d(4*init_channels,8*init_channels,4,2,1) 
       
        self.rb1 = ResidualBlock(8*init_channels,False,3,2,False)
        self.rb2 = ResidualBlock(8*init_channels,False,3,2,False)
        self.rb3 = ResidualBlock(8*init_channels,False,3,2,False)

        self.b3 = Block(inchannels=4*init_channels,outchannels=4*init_channels,dropout=False,kernel=3,bias=False,depth=2,mode='same',factor=2)
        self.b2 = Block(inchannels=2*init_channels,outchannels=2*init_channels,dropout=False,kernel=3,bias=False,depth=2,mode='same',factor=2)
        self.b1 = Block(inchannels=init_channels,outchannels=init_channels,dropout=False,kernel=3,bias=False,depth=2,mode='same',factor=2)

        ### decoder
        self.deconv41 = nn.ConvTranspose3d(8*init_channels,(8*init_channels)//2,4,2,1) 
        self.conv_u41 = nn.Conv3d(11*init_channels,(8*init_channels)//2,3,1,1)
        
        self.deconv31 = nn.ConvTranspose3d((8*init_channels)//2,(8*init_channels)//4,4,2,1) 
        self.conv_u31 = nn.Conv3d(9*init_channels,(8*init_channels)//4,3,1,1)
        
        self.deconv21 = nn.ConvTranspose3d((8*init_channels)//4,(8*init_channels)//8,4,2,1)
        self.conv_u21 = nn.Conv3d(8*init_channels,(8*init_channels)//8,3,1,1)
        
        self.deconv11 = nn.ConvTranspose3d((8*init_channels)//8,(8*init_channels)//16,4,2,1)
        self.conv_u11 = nn.Conv3d((8*init_channels)//16,(8*init_channels)//16,3,1,1)
        self.pool = nn.MaxPool3d(2, stride=2)


        ######################## second model
        self.conv1_main = nn.Conv3d(inc,init_channels,4,2,1) 
        self.conv2_main = nn.Conv3d(init_channels+(8*init_channels)//16,2*init_channels,4,2,1) 
        self.conv3_main = nn.Conv3d(2*init_channels+(8*init_channels)//8,4*init_channels,4,2,1)
        self.conv4_main = nn.Conv3d(4*init_channels+(8*init_channels)//4,8*init_channels,4,2,1) 
       
        self.rb1_main = ResidualBlock(8*init_channels+(8*init_channels)//2,False,3,2,False)
        self.rb2_main = ResidualBlock(8*init_channels+(8*init_channels)//2,False,3,2,False)
        self.rb3_main = ResidualBlock(8*init_channels+(8*init_channels)//2,False,3,2,False)

        self.b3_main = Block(inchannels=4*init_channels,outchannels=4*init_channels,dropout=False,kernel=3,bias=False,depth=2,mode='same',factor=2)
        self.b2_main = Block(inchannels=2*init_channels,outchannels=2*init_channels,dropout=False,kernel=3,bias=False,depth=2,mode='same',factor=2)
        self.b1_main = Block(inchannels=init_channels,outchannels=init_channels,dropout=False,kernel=3,bias=False,depth=2,mode='same',factor=2)


        # decoder 
        self.deconv41_main = nn.ConvTranspose3d(8*init_channels+(8*init_channels)//2,(8*init_channels)//2,4,2,1) 
        self.conv_u41_main = nn.Conv3d(11*init_channels,(8*init_channels)//2,3,1,1)
        
        self.deconv31_main = nn.ConvTranspose3d((8*init_channels)//2,(8*init_channels)//4,4,2,1) 
        self.conv_u31_main = nn.Conv3d(9*init_channels,(8*init_channels)//4,3,1,1)
        
        self.deconv21_main = nn.ConvTranspose3d((8*init_channels)//4,(8*init_channels)//8,4,2,1)
        self.conv_u21_main = nn.Conv3d(8*init_channels,(8*init_channels)//8,3,1,1)
        
        self.deconv11_main = nn.ConvTranspose3d((8*init_channels)//8,(8*init_channels)//16,4,2,1)
        self.conv_u11_main = nn.Conv3d((8*init_channels)//16,outc,3,1,1)


    def forward(self,x):
        # first model 
        x_pool = F.relu(self.input_pool(x)) 
        x1 = F.relu(self.conv1(x_pool)) 
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x4 = self.rb1(x4)
        x4 = self.rb2(x4)
        x4 = self.rb3(x4)

        x1 = self.b1(x1)
        x2 = self.b2(x2)
        x3 = self.b3(x3)


        # decoder
        u11 = F.relu(self.deconv41(x4))

        x1_pool1 = self.pool(x1)
        x1_pool2 = self.pool(x1_pool1)

        x2_pool1 = self.pool(x2)


        u11 = F.relu(self.conv_u41(torch.cat((u11,x3,x2_pool1,x1_pool2),dim=1)))




        u21 = F.relu(self.deconv31(u11)) # 2*init_channels
        x3_up1 = F.interpolate(x3, scale_factor=2.0, mode='trilinear', align_corners=True)

        u21 = F.relu(self.conv_u31(torch.cat((u21,x2,x1_pool1,x3_up1),dim=1)))




        u31 = F.relu(self.deconv21(u21))  # init_channels

        x3_up2 = F.interpolate(x3, scale_factor=4.0, mode='trilinear', align_corners=True)
        x2_up1 = F.interpolate(x2, scale_factor=2.0, mode='trilinear', align_corners=True)

        u31 = F.relu(self.conv_u21(torch.cat((u31,x1,x2_up1,x3_up2),dim=1)))
        


        u41 = F.relu(self.deconv11(u31))
        out = self.conv_u11(u41)


        #### second model 
        x1_main = F.relu(self.conv1_main(x)) 
        fuse1 = torch.cat([out, x1_main], 1) #(8*init_channels)//16+

        x2_main = F.relu(self.conv2_main(fuse1))
        fuse2 = torch.cat([u31, x2_main], 1) #

        x3_main = F.relu(self.conv3_main(fuse2))
        fuse3 = torch.cat([u21, x3_main], 1)

        x4_main = F.relu(self.conv4_main(fuse3))
        fuse4 = torch.cat([u11, x4_main], 1)

        x4_main = self.rb1_main(fuse4)
        x4_main = self.rb2_main(x4_main)
        x4_main = self.rb3_main(x4_main)


        x1_main = self.b1_main(x1_main)
        x2_main = self.b2_main(x2_main)
        x3_main = self.b3_main(x3_main)

        # decoder
        u11_main = F.relu(self.deconv41_main(x4_main))

        x1_pool1_main = self.pool(x1_main)
        x1_pool2_main = self.pool(x1_pool1_main)

        x2_pool1_main = self.pool(x2_main)


        u11_main = F.relu(self.conv_u41_main(torch.cat((u11_main,x3_main,x2_pool1_main,x1_pool2_main),dim=1)))




        u21_main = F.relu(self.deconv31_main(u11_main)) # 2*init_channels
        x3_up1_main = F.interpolate(x3_main, scale_factor=2.0, mode='trilinear', align_corners=True)

        u21_main = F.relu(self.conv_u31_main(torch.cat((u21_main,x2_main,x1_pool1_main,x3_up1_main),dim=1)))




        u31_main = F.relu(self.deconv21_main(u21_main))  # init_channels

        x3_up2_main = F.interpolate(x3_main, scale_factor=4.0, mode='trilinear', align_corners=True)
        x2_up1_main = F.interpolate(x2_main, scale_factor=2.0, mode='trilinear', align_corners=True)

        u31_main = F.relu(self.conv_u21_main(torch.cat((u31_main,x1_main,x2_up1_main,x3_up2_main),dim=1)))
        


        u41_main = F.relu(self.deconv11_main(u31_main))
        out_main = self.conv_u11_main(u41_main)

        return out_main



