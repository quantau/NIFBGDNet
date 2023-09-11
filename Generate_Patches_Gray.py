# import the libraries
import os
import argparse
import numpy as np
import cv2
from conf import myConfig_Gray as config
from pathlib import Path
from tqdm import tqdm

# define scales
scales=[1,0.9,0.8,0.7]
count=0

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir',
                    default='/home/anirban_2021cs13/Proj_RK/Dataset/Dataset', help='dir of data')
args = parser.parse_args()

# create the imageArrays
p=Path(config.genDataPath)
# listPaths=list(p.glob('./*.tif'))
listPaths = []

for root, _, files in tqdm(os.walk(args.src_dir)):
    for file in files:
        if file.endswith(".tif"):
            # Construct the full file path
            file_path = os.path.join(root, file)
            listPaths.append(file_path)

imgArray=[]
for path in listPaths:
    imgArray.append(cv2.imread(str(path),0))
print('lenImages',len(imgArray))

#calculate the number of patches
print("Calculating the number of patches")
for i in tqdm(range(len(imgArray))):
    img = imgArray[i] 
    for s in range(len(scales)):
        newsize=(int(img.shape[0]*scales[s]),int(img.shape[1]*scales[s]))
        img_s=cv2.resize(img,newsize,interpolation=cv2.INTER_CUBIC)
        im_h,im_w=img_s.shape
        for x in range(0+config.step,(im_h-config.pat_size),
                config.stride):
            for y in range(0+config.step,(im_w-config.pat_size),
                    config.stride):
                count +=1

origin_patch_num=count
if origin_patch_num % config.batch_size !=0:
    numPatches=(origin_patch_num/config.batch_size +1)*config.batch_size
else:
    numPatches=origin_patch_num
print('total patches=%d, batch_size=%d, total_batches=%d' % 
        (numPatches, config.batch_size, numPatches/config.batch_size))

#numpy array to contain patches for training
inputs=np.zeros((int(numPatches), int(config.pat_size), int(config.pat_size),1),dtype=np.uint8)

#generate patches
print("Genrating patches")
count=0
for i in tqdm(range(len(imgArray))):
    img=imgArray[i]
    for s in range(len(scales)):
        newsize=(int(img.shape[0]*scales[s]),int(img.shape[1]*scales[s]))
        img_s=cv2.resize(img,newsize,interpolation=cv2.INTER_CUBIC)
        img_s=np.reshape(np.array(img_s,dtype="uint8"),
                (img_s.shape[0],img_s.shape[1],1))
        im_h,im_w, _ = img_s.shape
        for x in range(0+config.step,im_h-config.pat_size,config.stride):
            for y in range(0+config.step,im_w-config.pat_size,
                config.stride):
                inputs[count,:,:,:]=img_s[x:x+config.pat_size,
                    y:y+config.pat_size,:]
                count += 1


#pad the batch
if count < numPatches:
    to_pad=int(numPatches-count)
    inputs[-to_pad:,:,:,:]=inputs[:to_pad,:,:,:]

if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
np.save(os.path.join(config.save_dir,"img_clean_pats"),inputs)
