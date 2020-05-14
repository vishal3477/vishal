import os
from PIL import Image
from function import *
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from xception import xception
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import argparse
import datetime
import pandas as pd
from tensorboardX import SummaryWriter



torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available() 
print(use_cuda)                  # check if GPU exists
#device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
device=torch.device('cuda:0')
torch.backends.deterministic = True

parser = argparse.ArgumentParser()
#parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='manual seed')
#parser.add_argument('--it_start', type=int, default=1, help='number of itr to start with')
#parser.add_argument('--it_end', type=int, default=10000, help='number of itr to end with')
parser.add_argument('--signature', default=str(datetime.datetime.now()))
# parser.add_argument('--data_dir', help='directory for data')
parser.add_argument('--save_dir', default='/mnt/gs18/scratch/users/asnanivi/runs', help='directory for result')
opt = parser.parse_args()
print(opt)

sig = str(datetime.datetime.now())
#os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
#os.makedirs('%s/modules/%s' % (opt.save_dir, sig), exist_ok=True)

CNN_embed_dim = 2048
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256
dropout_p = 0.0
k=2


print("Initializing Networks")
cnn_encoder = xception(2, load_pretrain=True).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
#optimizer_xcp = optim.Adam(model.parameters(), lr=opt.lr)
#model.cuda()
optimizer = torch.optim.Adam(crnn_params, lr=opt.lr)
cse_loss = nn.CrossEntropyLoss().cuda()

def train(batch, label):
    #model.train()
    cnn_encoder.train()
    rnn_decoder.train()
    temp=cnn_encoder(batch)
    temp=torch.unsqueeze(temp,0)
    temp=temp.permute(1,0,2)
    output = rnn_decoder(temp)
    loss = cse_loss(output, label.type(torch.cuda.LongTensor))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return[loss.item()]


def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)

main_dir="/mnt/scratch/asnanivi/frame_data/frame_data"
metadata_dir="/mnt/gs18/scratch/users/asnanivi/1"
writer = SummaryWriter('%s/logs/%s' % (opt.save_dir, sig))
#folder=next(os.walk(main_dir))
folders=["dfdc_train_part_14","dfdc_train_part_15","dfdc_train_part_16","dfdc_train_part_17"]
#onlyfiles = next(os.walk(os.path.join(main_dir,folder[0],folder[0],"Fake")))[2]
#print(len(onlyfiles))

state = {
    'state_dict_cnn':cnn_encoder.state_dict(),
    'state_dict_rnn': rnn_decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    
}


state1 = torch.load('/mnt/gs18/scratch/users/asnanivi/runs/logs/2020-03-24 20:51:46.488354/99000-dfdc_train_part_06.pickle')
rnn_decoder.load_state_dict(state1['state_dict_rnn'])
optimizer.load_state_dict(state1['optimizer'])
cnn_encoder.load_state_dict(state1['state_dict_cnn'])

for foldername in folders:
    imagefiles = next(os.walk(os.path.join(main_dir,foldername)))[2]
    imagefiles=sorted(imagefiles)
    metadata=pd.read_json(metadata_dir + '/'+foldername +"/metadata.json", orient='index')
    i=0
    prev_file=imagefiles[0]
    print(prev_file.split("_")[0])
    flag=0
    while i<len(imagefiles):
        j=0
        images=[]
        y_train=np.ones((30),dtype=float)
        while j<30 and i<len(imagefiles):
            images.append(cv2.imread(os.path.join(main_dir,foldername,imagefiles[i])))
            if imagefiles[i].split("_")[0]!=prev_file.split("_")[0] or flag==0:
                for index, row in metadata.iterrows():
                    if index.split(".")[0]==imagefiles[i].split("_")[0]:
                        if row[0]=="FAKE":
                            y_train[j]=0
                        elif row[0]=="REAL":
                            y_train[j]=1
                        y_prev=y_train[j]
                flag=1
               
            else:
                y_train[j]=y_prev
            print(imagefiles[i])
            if i%250==0: 
                print(imagefiles[i].split("_")[0],y_train[j])
            j=j+1
            i=i+1
        print(y_train)
        images = np.array(images)
        Y_train=np.array(y_train[0:j])
        X_train=torch.from_numpy(images).permute(0,3,1,2)
        X_train=X_train.byte()
        #print(X_train.shape)
        #train = torch.utils.data.TensorDataset(torch.from_numpy(images), torch.from_numpy(Y_train))
        loss = train(torch.from_numpy(images).permute(0,3,1,2), torch.from_numpy(Y_train))
        write_tfboard([loss[0]], i, name='TRAIN')
        if i % 900 == 0:
            torch.save(state, '%s/logs/%s/%d-%s.pickle' % (opt.save_dir, sig, i,foldername))
            print("Save Model: {:d}".format(i))
        
            
            
    