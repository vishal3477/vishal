import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from function import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import pandas as pd
import json
import glob
import time
import cv2
from matplotlib import pyplot as plt
# Detect devices
torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available() 
print(use_cuda)                  # check if GPU exists
#device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
device=torch.device('cuda:0')
class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames=30, batch_size=30, resize=1.4):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        #faces = []
        face=[]
        frames = []
        faces = []
        required_size=(224, 224)
        flag=0
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                #frames.append(frame)

                # When batch is full, detect faces and reset frame list
                #if len(frames) % self.batch_size == 0 or j == sample[-1]:
                
                face=self.detector(frame)
                
                if flag==0:
                    if face==None:
                        face=torch.zeros([3,256,342],dtype=torch.float32)
                        face_p=face
                    else:   
                        face=torch.unsqueeze(face,dim=0)
                        face = F.interpolate(face, size=(256,342), mode='bilinear', align_corners=False)
                        faces=face
                
                #print(face.shape)
                if face!=None and flag!=0:
                    face_p=face
                    face=torch.unsqueeze(face,dim=0)
                    face = F.interpolate(face, size=(256,342), mode='bilinear', align_corners=False)
                    faces = torch.cat([face,faces], dim=0)
                if face==None and flag!=0:
                    face=torch.unsqueeze(face_p,dim=0)
                    face = F.interpolate(face, size=(256,342), mode='bilinear', align_corners=False)
                    faces = torch.cat([face,faces], dim=0)
                flag=1
                
                #face=self.detector(frames)
                #faces.extend(face)
                #print(faces.shape)
                

                
        print(faces.shape)
        v_cap.release()

        return faces  
def process_faces(faces, resnet):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = resnet(faces)

    return embeddings

# set path
#data_path = "./mnt/gnt/users/asnanivi/facebook data/dfdc_train_part_0/"    # define UCF-101 RGB data path
save_model_path = "./CRNN_ckpt/"
DATA_FOLDER = '/mnt/gs18/scratch/users/asnanivi'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
#TEST_FOLDER = 'test_videos'

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512      # latent dim extracted by 2D CNN
img_x, img_y = 256, 342  # resize video 2d frame size
dropout_p = 0.0          # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
epochs = 10        # training epochs
batch_size = 30  
learning_rate = 1e-4
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score



#device = torch.device("cpu")

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
#print(params)

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

json_file = [file for file in train_list if  file.endswith('json')][0]
print(f"JSON file: {json_file}")

def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
meta_train_df.head()
filenames = glob.glob('/mnt/gs18/scratch/users/asnanivi/train_sample_videos/*.mp4')
print(len(filenames))

labels=[]
for fn in meta_train_df.index[:]:
    label = meta_train_df.loc[fn]['label']
    labels.append(label) 
    
    

action_names=["REAL","FAKE"] 
# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)
                
all_y_list = labels2cat(le, labels)    # all video labels
all_Y=torch.LongTensor(all_y_list)
print(all_Y.shape)


transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
# Load face detector
mtcnn = MTCNN(margin=14, keep_all=False, factor=0.5, device=device).eval()
# Define face detection pipeline
detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60, resize=0.25)

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
#resnet=InceptionV3(input_shape=img_shape, weights='imagenet', include_top=False, pooling='avg')
#X = torch.zeros([1334,30,3,256,342],dtype=torch.float64).to(device)
X=[]
#print(X.shape)
start = time.time()
n_processed = 0
flag1=0
with torch.no_grad():
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        try:
           
            # Load frames and find faces
            faces = detection_pipeline(filename)
            #X[i,:,:,:,:]=faces
            #waste=process_faces(faces, resnet)
            #print(waste.shape)
            #print(faces)
            # Calculate embeddings
            
            
            if flag1==0:
                faces=torch.unsqueeze(faces,dim=0)
                X=faces
                
            
            if flag1!=0:
                faces=torch.unsqueeze(faces,dim=0)
                X = torch.cat([faces,X], dim=0)
            
            flag1=1
            
            print(X.shape)
            
            
        except KeyboardInterrupt:
            print('\nStopped.')
            break

        except Exception as e:
            print(e)
            #X.append(None)
print("final shape")           
print(X.shape)             
train_set= X,all_Y
train_dataset = data.TensorDataset(X, all_Y)
#print(train_set)
#train_set= Variable(torch.from_numpy(X)),all_Y
                       
#valid_set=Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)
train_loader = data.DataLoader(train_dataset, **params)
#valid_loader = data.DataLoader(valid_set, **params)

#for batch_idx, (X, y) in enumerate(train_loader):
#    print(batch_idx)
#    print(X)
#    print(y)

# Create model
cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                         drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    print(epoch, "starting")
    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    
    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    print(epoch ,"done")
    print("loss", A)
    print("scores",B)
    np.save('/mnt/ufs18/home-188/asnanivi/Desktop/Deepfake/result', A)
    np.save('/mnt/ufs18/home-188/asnanivi/Desktop/Deepfake/result', B)
    

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_CRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()
