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

def upload_vqa(request):
    fs = FileSystemStorage()
    uploaded_file_url_vqa_hat = fs.url('gray.png')

     # default image
    att_image_url_san = fs.url('gray.png')
    vqa_data_file = {}     # default image
    if request.method == 'POST':
        imgs_train = json.load(open('/home/ksharan1/visualization/san-vqa-tensorflow/data/vqa_raw_train.json', 'r'))
        list_of_paths=[]
        list_of_img=[]
        index=random.randint(0,45000)
        print(index)
        print(imgs_train[index])

        list_of_img=[]
        for x in imgs_train:
            list_of_img.append(x['ques_id'])

        print(list_of_img)


        vqa_data_file = {}
        pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/vqa-hat/vqahat_train"


        for image_path in os.listdir(pathdir):
            input_path = os.path.join(pathdir, image_path)
            print(input_path)

            question_id=image_path.split('_')[0]
            for x in range(0,len(list_of_img)):
                print("1")
                print(imgs_train[index]['ques_id'])
                print("2")
                print(str(list_of_img[x]))

                if str(list_of_img[x])==imgs_train[index]['ques_id']:
                    uploaded_file_url_vqa_hat=input_path
                    break


        vqa_data_file['image'] ="../../"+imgs_train[index]['img_path']
        vqa_data_file['question'] = imgs_train[index]['question']
        print(vqa_data_file['image'])
        print(vqa_data_file['question'])


        json.dump(vqa_data_file, open('/home/ksharan1/visualization/san-vqa-tensorflow/demo/vqa_data_file.json', 'w'))
        pathdir = "/home/ksharan1/san-vqa-tensorflow/vqa-hat/vqahat_train"


        image = vqa_data_file['image']
        question = vqa_data_file['question']
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)
        vqa_data['image'] = uploaded_file_url
        vqa_data['question'] = question
        print
            # save uploaded question-image pair
        json.dump(vqa_data, open('vqa_data.json', 'w'))
        answer = ''
            # wait for processed result
        while 1:
            if fs.exists('vqa_ans.txt'):
                with fs.open('vqa_ans.txt', 'r') as f:
                    answer = f.read()
                    att_image_url_san = fs.url('att.jpg')
                    fs.delete('vqa_ans.txt')

            break
        return render(request, 'upload.html', {
                'uploaded_file_url': uploaded_file_url_vqa_hat,
                'att_image_url': att_image_url_san,
                'input_question': question,
                'vqa_ans': answer

            })
    return render(request, 'upload.html', {
                'uploaded_file_url': uploaded_file_url_vqa_hat,
                'att_image_url': att_image_url_san
            })
