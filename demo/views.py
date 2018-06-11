from django.shortcuts import render
from django.http import HttpResponse, HttpResponseForbidden
from django.core.files.storage import FileSystemStorage
import json
import random
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json
import os.path
import re
from django.core.files import File
import time


def upload_vqa(request):
    fs = FileSystemStorage()
    uploaded_file_url = fs.url('gray.png')

     # default image

    att_image_url = fs.url('gray.png')
    vqa_data_file = {}     # default image
    if request.method == 'POST':
        imgs_train = json.load(open('/home/ksharan1/visualization/san-vqa-tensorflow/data/vqa_raw_train.json', 'r'))
        list_of_paths=[]
        list_of_img=[]
        list_of_img=[]
        list_of_vqa_paths=[]
        list_vqa_hat=[]
        for x in imgs_train:
            list_of_img.append(x['ques_id'])
        vqa_data_file = {}
        pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/demo/media"
        for image_path in os.listdir(pathdir):
            input_path = os.path.join(pathdir, image_path)
            question_id=image_path.split('_')[0]
            list_vqa_hat.append(question_id)
            list_of_vqa_paths.append(image_path)
        index=random.randint(0,len(list_vqa_hat))
        for x in range(len(list_of_img)-1):
            input_path = fs.url(str(list_of_vqa_paths[index]))

            if str(list_of_img[x])==str(list_vqa_hat[index]):
                uploaded_file_url=input_path
                break

        vqa_data_file['image'] ="/home/ksharan1/visualization/san-vqa-tensorflow/data/"+imgs_train[index]['img_path']

        originalimage=imgs_train[index]['img_path'].split('/')[1]
        print originalimage
        originalimage=fs.url(str(originalimage))
        vqa_data_file['question'] = imgs_train[index]['question']
        json.dump(vqa_data_file, open('/home/ksharan1/visualization/san-vqa-tensorflow/demo/vqa_data_file.json', 'w'))
        pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/demo/media"
        image = vqa_data_file['image']
        question = vqa_data_file['question']


        json.dump(vqa_data_file, open('vqa_data.json', 'w'))
        answer = ''
        time.sleep(5)
        while 1:
            if fs.exists('vqa_ans.txt'):

                with fs.open('vqa_ans.txt', 'r') as f:
                    #time.sleep(5)
                    answer = f.read()
                    print answer

                    att_image_url = fs.url('att.jpg')
                    fs.delete('vqa_ans.txt')
            break
        #time.sleep(2)
        return render(request, 'upload.html', {
                'uploaded_file_url': uploaded_file_url,
                'att_image_url': att_image_url,


                'input_question': question,
                'vqa_ans': answer

            })
    time.sleep(2)
    return render(request, 'upload.html', {
                'uploaded_file_url': uploaded_file_url,

                'att_image_url': att_image_url
            })
