import scipy.io
#import cv2
import h5py
import glob
import os
import numpy as np

f=open('../traindata/nyu_40/traindata_hdf5/trainlist.txt','w')
def savehdf5(path):
	dic=scipy.io.loadmat(path)
	name=path.split('/')[-1].split('.')[0]
	
	print name
	a=dic['image']
	b=dic['depth']
	c=dic['point']
	d=dic['label']
        IMAGE=np.zeros((1,3, a.shape[0], a.shape[1]))
	IMAGE[0,0,:,:] = a[:,:,0]
        IMAGE[0,1,:,:] = a[:,:,1]
        IMAGE[0,2,:,:] = a[:,:,2]
        
        DEPTH=np.zeros((1,1,a.shape[0],a.shape[1]))
	DEPTH[0,0,:,:]=b

        POINT = np.zeros((1,3,a.shape[0], a.shape[1]))
        POINT[0,0,:,:] = c[:,:,0]
        POINT[0,1,:,:] = c[:,:,1]
        POINT[0,2,:,:] = c[:,:,2]

	LABEL=np.zeros((1,1,a.shape[0],a.shape[1]))
	LABEL[0,0,:,:]=d
	
	with h5py.File('../traindata/nyu_40/traindata_hdf5/'+name+'.h5', 'w') as hdf5file:
	    hdf5file['image'] = IMAGE
            hdf5file['depth']= DEPTH
            hdf5file['point'] = POINT
            hdf5file['label'] = LABEL
        
paths=glob.glob('../traindata/nyu_40/traindata_mat/*.mat')
print 'generate'
fwrong=open('../traindata/nyu_40/traindata_hdf5/buglist_data.txt','w')
for path in paths:
  
  name=path.split('/')[-1].split('.')[0]
  
  
  
  try:
    #if os.path.isfile('/home/xjqi/Train/lzz/hdf5/'+name+'.h5'):
    #	continue
  	savehdf5(path)
        f.write('../traindata/nyu_40/traindata_hdf5/'+name+'.h5\n')
    
  except:
  	fwrong.write(name+'\n')
fwrong.close()
f.close()
