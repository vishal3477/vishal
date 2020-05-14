import numpy as np
import pandas as pd
import cv2
import os
#from RetinaFace.retinaface import RetinaFace
#from RetinaFace.face_align import norm_crop
from torchvision import transforms
import random
import torch
import pickle
from facenet_pytorch import MTCNN
import glob
from PIL import Image


mtcnn = MTCNN(select_largest=False)
metadatas=[]
data_dir="/mnt/gs18/scratch/users/asnanivi/1/"
#print(data_dir)

folders=["dfdc_train_part_08","dfdc_train_part_09","dfdc_train_part_10","dfdc_train_part_11","dfdc_train_part_12","dfdc_train_part_00"]
#metadatas=pd.read_json(data_dir  + "/deepfake-detection-challenge/"+folders[0]+"/metadata.json", orient='index')
for f in folders:
    metadatas.append([pd.read_json(data_dir +f + "/metadata.json", orient='index'), f])
n_frames=30
for metadata in metadatas:
    for foldername in folders:
        #filenames = glob.glob('C:/Users/visha/Desktop/MSU/Prof. Liu/deepfake-detection-challenge/'+foldername+'/*.mp4')
        for index, row in metadata[0].iterrows():
            v_cap = cv2.VideoCapture(data_dir+foldername+"/"+index)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(v_len)
            # Loop through video
            batch_size = 32
            frames = []
            boxes = []
            landmarks = []
            view_frames = []
            view_boxes = []
            view_landmarks = []
            sample = np.linspace(0, v_len - 1, n_frames).astype(int)
            for count in range(v_len):
                if count in sample:
                    # Load frame
                    success, frame = v_cap.read()
                    if not success:
                        continue

                    # Add to batch, resizing for speed
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = frame.resize([int(f * 0.25) for f in frame.size])
                    #frames.append(frame)
                    out_name = "{:s}_{:0>4d}.png".format(index.split('.')[0], count)
                    #save_paths = ["D:/facebook_frames/"+foldername+"/"+'image_{i}.jpg' for i in range(v_len)]
                    mtcnn(frame, save_path="/mnt/home/asnanivi/Desktop/Deepfake/"+foldername+"/"+out_name)


