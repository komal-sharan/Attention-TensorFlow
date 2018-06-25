import pickle
import os, h5py, sys, argparse
import numpy as np
listforme=[]
train_data = {}
vqa_data={}
data=[]
input_ques_h5 = '/home/ksharan1/san-vqa-tensorflow/data_prepro.h5'
pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/vqa-hat/vqahat_train"
count = 0
path={}
pathlist={}

with h5py.File(input_ques_h5,'r') as hf:
    tem = hf.get('question_id_train')
    train_data['ques_id'] = np.array(tem)
print "done"
for image_path in os.listdir(pathdir):
    question_id=image_path.split('_')[0]
    path[question_id]=image_path

print "done"
for x in range(len(train_data['ques_id'])):
    if str(train_data['ques_id'][x]) in path.keys():
        vqa_data['index']=x
        vqa_data['filename']=path[question_id]
        data.append(vqa_data)
        count=count+1
        print count





pickle.dump(data, open("save2.pkl", "wb"))
