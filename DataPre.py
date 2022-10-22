import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os


class DataSet():
	def __init__(self,args):
		self.path = args.data_path
		self.dataset = args.dataset
		if self.dataset == 'supercurrent':
			self.dim = [256,128,32]
			self.c = [256,128,32]
			self.total_samples = 100
		elif self.dataset == 'tornado':
			self.dim = [128,128,128]
			self.c = [128,128,128]
			self.total_samples = 48
		elif self.dataset == 'hurricane':
			self.dim = [500,500,100]
			self.c = [256,224,32]
			self.total_samples = 48
		elif self.dataset == 'supernova':
			self.dim = [128,128,128]
			self.c = [128,128,128]
			self.total_samples = 100
		elif self.dataset == 'cylinder':
			self.dim = [640,240,80]
			self.c = [80,80,80]
			self.total_samples = 100
		self.training_samples = args.samples
		self.vec = np.zeros((self.total_samples,3,self.dim[0],self.dim[1],self.dim[2]))
		self.scalar = np.zeros((self.total_samples,1,self.dim[0],self.dim[1],self.dim[2]))
		self.croptimes = args.croptimes
		if (self.dim[0] == self.c[0]) and (self.dim[1] == self.c[1]) and (self.dim[2] == self.c[2]):
			self.croptimes = 1

	def ReadData(self):
		for i in range(1,self.total_samples+1):
			print(i)
			path = '/afs/crc.nd.edu/group/vis/Vector/V/supercurrent/'+'{:03d}'.format(9*i)+ '.vec'
			#print('path', path)
			v = np.fromfile(path,dtype='<f')
			v = v.reshape(self.dim[2],self.dim[1],self.dim[0],3).transpose()
			self.vec[i-1] = v
			scalar = np.sqrt(np.sum(v**2,axis=0))
			self.scalar[i-1][0] = 2*(scalar-np.min(scalar))/(np.max(scalar)-np.min(scalar))-1

	def GetTrainingData(self):
		s = np.zeros((self.croptimes*self.training_samples,1,self.c[0],self.c[1],self.c[2]))
		v = np.zeros((self.croptimes*self.training_samples,3,self.c[0],self.c[1],self.c[2]))
		idx = 0
		sampleList = [65, 61, 35, 44, 23, 
					82, 36, 83, 67, 93, 
					50, 15, 28, 62, 99, 
					46, 55, 29, 96, 92, 
					69, 95, 90, 6, 51, 
					38, 78, 53, 33, 7, 
					48, 34, 73, 88, 39, 
					64, 76, 72, 87, 56]
		#print('sampleList', sampleList)
		for k in sampleList:
			sc, vc = self.CropData(self.scalar[k],self.vec[k])
			for j in range(0,self.croptimes):
				s[idx] = sc[j]
				v[idx] = vc[j]
				idx += 1
		s = torch.FloatTensor(s)
		v = torch.FloatTensor(v)
		data = torch.utils.data.TensorDataset(s,v)
		train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
		return train_loader

	def CropData(self,s,v):
		sc = []
		vc = []
		n = 0
		while n<self.croptimes:
			if self.c[0]==self.dim[0]:
				x = 0
			else:
				x = np.random.randint(0,self.dim[0]-self.c[0])
			if self.c[1] == self.dim[1]:
				y = 0
			else:
				y = np.random.randint(0,self.dim[1]-self.c[1])
			if self.c[2] == self.dim[2]:
				z = 0
			else:
				z = np.random.randint(0,self.dim[2]-self.c[2])
			sc_ = s[0:1,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			vc_ = v[0:3,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			sc.append(sc_)
			vc.append(vc_)
			n = n+1
		return sc,vc


