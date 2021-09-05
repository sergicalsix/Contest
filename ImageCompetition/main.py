!pip3 install timm
!pip3 install natsort
! pip install albumentations

import os
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torchvision
import torchvision.utils as vutils
import torchvision.datasets as dset
from  torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import timm


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    

SEED = 42
fix_seed(SEED)

import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

train_path_= glob.glob('./ImageClustering2/train/0/*')
train_path = glob.glob('./ImageClustering2/train/1/*')
train_path_2 = glob.glob('./ImageClustering2/train/2/*')
train_path_3 = glob.glob('./ImageClustering2/train/3/*')
path = glob.glob('./ImageClustering2/test/*')


im = cv2.imread(train_path[84]) / 255
im = cv2.resize(im, dsize=(256, 256))
#im = im[64 : 192 , 64 : 192]
plt.axis('off')
plt.imshow(im)

im = cv2.imread(path[80]) / 255
im = cv2.resize(im, dsize=(256, 256))
im2 = cv2.GaussianBlur(im, (7, 7),0)
plt.imshow(im2)


model_name = 'efficientnet_b4'
#GPUID
ngpu = 1
batch_size = 10
#学習率
lr = 0.001
#Adamのbeta1
beta1 = 0.9


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class EfficientNet_b4(nn.Module):
    def __init__(self, n_out):
        super(EfficientNet_b4, self).__init__()
        self.effnet = timm.create_model('efficientnet_b4', pretrained=True)
        self.effnet.classifier = nn.Linear(1792, n_out) # bad code

    def forward(self, x):
        return self.effnet(x)
class HRNet(nn.Module):
    def __init__(self, n_out):
        super(HRNet, self).__init__()
        self.effnet = timm.create_model('vit_base_patch16_224', pretrained=True)
        #self.effnet.classifier = nn.Linear(2048, n_out) # bad code
        self.effnet.classifier = nn.Linear(2304, n_out) # bad code

    def forward(self, x):
        return self.effnet(x)
import cv2
train_x = []
train_y = []
n_label = 5
batch_size = 6
max_file_num = 840
img_size = 256 #256

for i in range(n_label):
    path = './ImageClustering2/train/'+str(i)+ '/*'
    files_path = glob.glob(path)
    file_count= 0 
    tmp_img = []
    tmp_idx = 0
    blur_img = 0
    print(len(files_path)) #max 147 min 84
    for file_path in files_path:
        if file_count < max_file_num:
            img = cv2.imread(file_path) / 255
            img = cv2.resize(img, dsize=(img_size, img_size)) # crop
            #img = cv2.normalize(img,None , norm_type=cv2.NORM_MINMAX)
            #img = img[64 : 192 , 64 : 192]
            train_x.append(img)
            train_y.append(i)
            tmp_img.append(img)
            file_count +=1
    while(file_count < 160):
        img = cv2.flip(tmp_img[tmp_idx], 0)
        train_x.append(img)
        train_y.append(i)
        file_count+=1
        tmp_idx+=1
    j = 0
    while(j < 80):
        im2 = cv2.GaussianBlur(tmp_img[j], (7, 7),0)
        train_x.append(img)
        train_y.append(i)
        j+=1
    
    
            
            
       
        

        """
        if i == 0:
            train_y.append([1,0,0,0,0])
        elif i == 1:
            train_y.append([0,1,0,0,0])
        elif i == 2:
            train_y.append([0,0,1,0,0])
        elif i == 3:
            train_y.append([0,0,0,1,0])
        else:
            train_y.append([0,0,0,0,1])"""
    
X, Y = np.array(train_x), np.array(train_y)
#X = (X - X.mean())/ X.std()
X_ = X.transpose(0,3,1, 2)#[N, C, W, H]
print(X_.shape,Y.shape) # ((575, 3, 512, 512), (575, 5) -> (575,))

from mysrc.dataset import DataSet
from sklearn.model_selection import train_test_split

from natsort import natsorted
test_x = []
n_label = 5
test_size = 2307

for _ in range(1):
    path = './ImageClustering2/test/*'
    files_path = glob.glob(path)
    files_path = natsorted(files_path)
    #print(files_path[:5])
    for img_path in files_path:
        img = cv2.imread(img_path) / 255
        img = cv2.resize(img, dsize=(256, 256)) # crop
        #img = img[64 : 192 , 64 : 192]
        test_x.append(img)
       
        """
        if i == 0:
            train_y.append([1,0,0,0,0])
        elif i == 1:
            train_y.append([0,1,0,0,0])
        elif i == 2:
            train_y.append([0,0,1,0,0])
        elif i == 3:
            train_y.append([0,0,0,1,0])
        else:
            train_y.append([0,0,0,0,1])"""
    
test_x = np.array(test_x)
#test_x = (test_x - X.mean()) / X.std()
test_x = test_x.transpose(0,3,1,2)
test_x.shape # 

from sklearn.model_selection import KFold
from statistics import mean

# 分割前のデータとして配列等をSeriesあるいはDataFrameに変換している
# fold毎のモデル、正解率、損失を格納する配列
nets, accs, losses = [], [], []

# n_splits分割（ここでは5分割）してCV
kf = KFold(n_splits=5, shuffle=True, random_state=2020)
## learning
SEED = 2022
fix_seed(SEED)
epochs = 20
jj = 1

for train_idx, valid_idx in kf.split(X_):
    print(jj)
    jj += 1
    # train_idx、valid_idxにはインデックスの値が格納されている
    train_x = X_[train_idx]
    train_y = Y[train_idx]
    val_x = X_[valid_idx]
    val_y = Y[valid_idx]
    
    #print(train_x.shape,train_y.shape)
    #print(val_x.shape,val_y.shape)
    
    # train_one関数では実際の訓練が行われる
    train_dataset = DataSet(x = train_x, t = train_y, transform = False, mixup = False)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,drop_last = True  )
    val_dataset = DataSet(x = val_x, t = val_y)
    val_loader = DataLoader(val_dataset, batch_size = 3, shuffle = True,drop_last = True  )
    
    model = EfficientNet_b4(n_label)
    #model = HRNet(n_label)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)

    
    dataloaders_dict = {"train":train_loader, "val":val_loader}
    accuracy, accuracy_val = [], []
    model.to(device)
    #model.to("cuda:0")
    for epoch in range(epochs):

        for phase in ["train", "val"]:
            #print("hoge")
            if phase == "train":
                model.train()
            else:
                model.eval()

            loss_epoch = 0.0
            acc_epoch = 0.0

            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in dataloaders_dict[phase]:
                optimizer.zero_grad()
                if inputs.shape[1] != 3:
                    inputs = inputs.permute(0,2,1,3)

                with torch.set_grad_enabled(phase == "train"):
                    labels = labels.to(device)
                    inputs= inputs.float() #Input type (torch.cuda.DoubleTensor
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    #print(outputs)

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    loss_epoch += loss.item() * inputs.size(0)
                    acc_epoch += torch.sum(preds == labels.data)

            loss_epoch = loss_epoch / len(dataloaders_dict[phase].dataset)
            acc_epoch = acc_epoch.double() / len(dataloaders_dict[phase].dataset)
            print(f"phase: {phase}",
                 # f"LR: {scheduler.get_last_lr()[0]} "
                  f"epoch: {epoch}",
                  f"loss: {loss_epoch:.4f}",
                  f"accuracy: {acc_epoch:.4f}")
                 # f"accuracy: {acc_epoch}:.4f}")
            if phase == "train":
                scheduler.step()

    print('Finished Training')
    nets.append(model)

test_x = []
n_label = 5
test_size = 2307

for _ in range(1):
    path = './ImageClustering2/test/*'
    files_path = glob.glob(path)
    files_path = natsorted(files_path)
    #print(files_path[:5])
    for img_path in files_path:
        img = cv2.imread(img_path) / 255
        img = cv2.resize(img, dsize=(256, 256)) # crop
        flag = False
        
        # tatesenn detect
        t_im = cv2.imread(img_path)
        t_im = cv2.resize(t_im, dsize=(256, 256))
        ret2, t_im = cv2.threshold(t_im, 160, 255,cv2.THRESH_BINARY)
        
        for i in range(256):
            count = 0 
            for j in range(10):
                if t_im[i][j][0] == 255:
                    count += 1          
            if count > 8 :
                flag = True

        #if flag:
         #   test_x.append(np.rot90(img))
        #else:
         #   test_x.append(img)
        test_x.append(img)
       
        """
        if i == 0:
            train_y.append([1,0,0,0,0])
        elif i == 1:
            train_y.append([0,1,0,0,0])
        elif i == 2:
            train_y.append([0,0,1,0,0])
        elif i == 3:
            train_y.append([0,0,0,1,0])
        else:
            train_y.append([0,0,0,0,1])"""
    
test_x = np.array(test_x)
#test_x = (test_x - X.mean()) / X.std()
test_x = test_x.transpose(0,3,1,2)
test_x.shape 

test_input = torch.from_numpy(test_x)
test_input  = test_input.float()

all_out = []

with torch.no_grad():
    for i in range(5):
        if i == 0 or i == 1 or i ==3:
            net = nets[i]
            out = []
            for one_input in test_input:
                #print("hoge")
                one_input = one_input.reshape(1, 3 , img_size , img_size)
                one_input = one_input.to(device)
                o = net(one_input)
                value =  o.cpu().numpy() 
                out.append(value.argmax())
            all_out.append(out)
            
d = np.array(all_out)
ans = []
count = 0
for i in range(2307):
    uniqs, counts = np.unique(d[:,i], return_counts=True)   
    val = uniqs[counts == np.amax(counts)]
    
    if val.shape ==(2,):
        count +=1
        #print(i)
        val = val[1]
    if val.shape ==(3,):
        count +=1
        #print(i)
        val = val[1]
    if val.shape ==(4,):
        count +=1
        #print(i)
        val = val[3]
    if val.shape ==(5,):
        count +=1
        #print(i)
        val = val[3]
    ans.append(int(val))
print(len(ans), count)

with open('result/sub17.csv', mode= 'w') as f:
    f.write('label\n')
    for val in ans:
        f.write(str(val))
        f.write('\n')

