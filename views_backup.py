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
import cv2
import numpy as np
from PIL import Image


def testoverlay(image1,image2):
    fs = FileSystemStorage()
    background = Image.open(image1)

    overlay = Image.open(image2)
    background = background.convert("RGBA")
    over=cv2.imread(image2).astype(np.float32)
    back=cv2.imread(image1).astype(np.float32)
    print over.shape
    print back.shape
    print back.shape[0]
    height, width = back.shape[:2]
    print height
    print width


    resizeoverlay=cv2.resize(over,(width,height),interpolation = cv2.INTER_CUBIC)
    #resizeoverlay.save("/home/ksharan1/visualization/san-vqa-tensorflow/new1.png","PNG")
    #resizeoverlayurl=fs.url(fs.save("newoverlay.png",resizeoverlay))
    cv2.imwrite("/home/ksharan1/visualization/san-vqa-tensorflow/new1.png", resizeoverlay)
    overlay=Image.open("/home/ksharan1/visualization/san-vqa-tensorflow/new1.png")

    print cv2.imread("/home/ksharan1/visualization/san-vqa-tensorflow/new1.png").astype(np.float32).shape
    print cv2.imread(image1).astype(np.float32).shape
    overlay = overlay.convert("RGBA")

    new_img = Image.blend(background, overlay, 0.9)
    new_img.save("/home/ksharan1/visualization/san-vqa-tensorflow/demo/media/new.png","PNG")
    return fs.url("new.png")





def attention_comb(original, vqahat):
    orig=cv2.imread(original).astype(np.float32)
    vqa=cv2.imread(vqahat).astype(np.float32)
    new_img = vqa*255*0.8 + orig*0.2
    return new_img


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
        inputp=""
        for image_path in os.listdir(pathdir):
            input_path = os.path.join(pathdir, image_path)
            inputp=os.path.join(pathdir, image_path)
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

        originalimage=fs.url(str(originalimage))
        vqa_data_file['question'] = imgs_train[index]['question']
        json.dump(vqa_data_file, open('/home/ksharan1/visualization/san-vqa-tensorflow/demo/vqa_data_file.json', 'w'))
        pathdir = "/home/ksharan1/visualization/san-vqa-tensorflow/demo/media"
        image = vqa_data_file['image']
        question = vqa_data_file['question']
        #new_img=attention_comb(vqa_data_file['image'],uploaded_file_url)
        #uploaded_file_url=fs.url(fs.save("new.png",new_img))
        print "imhere"


        json.dump(vqa_data_file, open('vqa_data.json', 'w'))
        answer = ''
        time.sleep(5)
        while 1:
            if fs.exists('vqa_ans.txt'):
                with fs.open('vqa_ans.txt', 'r') as f:
                    #time.sleep(5)
                    answer = f.read()
                    #print answer
                    print "fvgfgbtb"
                    #print vqa_data_file['question']
                    uploaded_file_url=testoverlay(vqa_data_file['image'],inputp)
                    #uploaded_file_url="/home/ksharan1/visualization/san-vqa-tensorflow/new.png"
                    #print fs.url(new_img.save("new.png","PNG"))
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
