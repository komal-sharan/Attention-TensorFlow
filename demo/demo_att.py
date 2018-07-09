from nltk.tokenize import word_tokenize
from scipy.ndimage.filters import gaussian_filter
import scipy.misc as misc
import numpy as np
import cv2
import caffe
import numpy as np
import tensorflow as tf
import json
import os
import time
import pdb
from PIL import Image

rnn_cell = tf.contrib.rnn
vars_to_rename = {
    "embed/multi_rnn_cell/cell_0/lstm_cell/weights": "embed/multi_rnn_cell/cell_0/lstm_cell/kernel",
    "embed/multi_rnn_cell/cell_1/lstm_cell/weights": "embed/multi_rnn_cell/cell_1/kernel",
    "embed/multi_rnn_cell/cell_0/lstm_cell/biases": "embed/multi_rnn_cell/cell_0/lstm_cell/bias",
    "embed/multi_rnn_cell/cell_1/lstm_cell/biases": "embed/multi_rnn_cell/cell_1/lstm_cell/bias",
}

class Answer_Generator():
    def __init__(self, rnn_size, rnn_layer, batch_size, input_embedding_size, dim_image, dim_hidden, dim_attention, max_words_q, vocabulary_size, drop_out_rate, sess):

        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.input_embedding_size = input_embedding_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.dim_att = dim_attention
        self.max_words_q = max_words_q
        self.vocabulary_size = vocabulary_size
        self.drop_out_rate = drop_out_rate
        self._sess = sess

        self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image[0], self.dim_image[1], self.dim_image[2]])
        self.question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])

        # question-embedding
        self.embed_ques_W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08), name='embed_ques_W')
        # encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
        #self.lstm_1 = rnn_cell.LSTMCell(rnn_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = 1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)

        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, use_peepholes=True)

        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = 1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # image-embedding
        self.embed_image_W = tf.Variable(tf.random_uniform([self.dim_image[2], self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_image_b')
        # score-embedding
        self.embed_scor_W = tf.Variable(tf.random_uniform([self.dim_hidden, num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -0.08, 0.08), name='embed_scor_b')
        print "newest"
        # Attention weight
        # question-attention
        #self.ques_att_W = tf.Variable(tf.random_uniform([self.dim_hidden, self.dim_att], -0.08, 0.08),name='ques_att_W')

        #self.ques_att_b = tf.Variable(tf.random_uniform([self.dim_att], -0.08, 0.08), name='ques_att_b')
        # image-attention
        #self.image_att_W = tf.Variable(tf.random_uniform([self.dim_hidden, self.dim_att], -0.08, 0.08),name='image_att_W')
        #self.image_att_b = tf.Variable(tf.random_uniform([self.dim_att], -0.08, 0.08), name='image_att_b')
        # probability-attention
        #self.prob_att_W = tf.Variable(tf.random_uniform([self.dim_att, 1], -0.08, 0.08),name='prob_att_W')
        #self.prob_att_b = tf.Variable(tf.random_uniform([1], -0.08, 0.08), name='prob_att_b')

    def build_generator(self):
        self.image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image[0], self.dim_image[1], self.dim_image[2]])
        self.question = tf.placeholder(tf.int32, [self.batch_size, self.max_words_q])

        state = self.stacked_lstm.zero_state(self.batch_size, tf.float32)
        loss = 0.0
        states_feat=[]
        with tf.variable_scope("embed"):
            for i in range(max_words_q):
               if i==0:
                   ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
                   print "given shape"
                   print ques_emb_linear.shape
               else:
                   tf.get_variable_scope().reuse_variables()
                   ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W,self.question[:,i-1])
                   print "now"
                   print ques_emb_linear.shape


               ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)


               ques_emb = tf.tanh(ques_emb_drop)

               output, state = self.stacked_lstm(ques_emb, state)

               #states_feat.append(ques_emb_linear)
            #question_feat = tf.stack(states_feat,axis=-1)
            #question_emb = state
        question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])

        # multimodal (fusing question & image)
        #question_emb = state
        #question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])

        image_emb = tf.reshape(self.image, [-1, self.dim_image[2]]) # (b x m) x d
        image_emb = tf.nn.xw_plus_b(image_emb, self.embed_image_W, self.embed_image_b)
        image_emb = tf.tanh(image_emb)

        #attention models
        with tf.variable_scope("att1"):
            self.prob_att1, comb_emb = self.attention(question_emb, image_emb)
        with tf.variable_scope("att2"):
            self.prob_att2, comb_emb = self.attention(comb_emb, image_emb)
        comb_emb = tf.nn.dropout(comb_emb, 1 - self.drop_out_rate)
        scores_emb = tf.nn.xw_plus_b(comb_emb,self.embed_scor_W, self.embed_scor_b)

        # FINAL ANSWER
        self.generated_ANS = tf.nn.softmax(scores_emb)

    def attention(self, question_emb, image_emb):

        ques_att_W = tf.get_variable('ques_att_W', [self.dim_hidden, self.dim_att],
                           initializer=tf.random_uniform_initializer(-0.08, 0.08))
        ques_att_b = tf.get_variable('ques_att_b', [self.dim_att],
                           initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # image-attention
        image_att_W = tf.get_variable('image_att_W', [self.dim_hidden, self.dim_att],
                           initializer=tf.random_uniform_initializer(-0.08, 0.08))
        image_att_b = tf.get_variable('image_att_b', [self.dim_att],
                           initializer=tf.random_uniform_initializer(-0.08, 0.08))
        # probability-attention
        prob_att_W = tf.get_variable('prob_att_W', [self.dim_att, 1],
                           initializer=tf.random_uniform_initializer(-0.08, 0.08))
        prob_att_b = tf.get_variable('prob_att_b', [1],
                            initializer=tf.random_uniform_initializer(-0.08, 0.08))
        question_att = tf.expand_dims(question_emb, 1) # b x 1 x d
        #question_att.shape
        #question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1])) # b x m x d
        #question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0]  * self.dim_image[1], 1])) # b x m x d

        question_att = tf.reshape(question_att, [-1, self.dim_hidden]) # (b x m) x d
        question_att = tf.tanh(tf.nn.xw_plus_b(question_att, ques_att_W, ques_att_b)) # (b x m) x k

        image_att = tf.tanh(tf.nn.xw_plus_b(image_emb, image_att_W, image_att_b)) # (b x m) x k

        output_att = tf.tanh(image_att + question_att) # (b x m) x k
        output_att = tf.nn.dropout(output_att, 1 - self.drop_out_rate)
        prob_att = tf.nn.xw_plus_b(output_att, prob_att_W, prob_att_b) # (b x m) x 1
        prob_att = tf.reshape(prob_att, [self.batch_size, self.dim_image[0] * self.dim_image[1]]) # b x m
        prob_att = tf.nn.softmax(prob_att)

        image_att = []
        image_emb = tf.reshape(image_emb, [self.batch_size, self.dim_image[0] * self.dim_image[1], self.dim_hidden]) # b x m x d
        for b in range(self.batch_size):
            image_att.append(tf.matmul(tf.expand_dims(prob_att[b,:],0), image_emb[b,:,:]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.add(image_att, question_emb)

        return prob_att, comb_emb

    def test(self, image, question):

        generated_ans, prob_att1, prob_att2 = self._sess.run(
            [self.generated_ANS, self.prob_att1, self.prob_att2],
            feed_dict={
                self.image: image,
                self.question: question
            })
        return generated_ans, prob_att1, prob_att2

##########################################################
web_path = '/home/ksharan1/visualization/san-vqa-tensorflow/demo/'

vqa_data_json = os.path.join(web_path, 'vqa_data.json')
answer_txt = os.path.join(web_path, 'media/vqa_ans.txt')
att_image = os.path.join(web_path, 'media/att.jpg')
#ques = 'What is this photo taken looking through?'


itoa_json = '/home/ksharan1/visualization/san-vqa-tensorflow/ix_to_ans.json'            # mapping idx to answer
wtoi_json = '/home/ksharan1/visualization/san-vqa-tensorflow/word_to_ix.json'           # mapping word to idx
cnn_proto = '/home/ksharan1/visualization/san-vqa-tensorflow/vgg19/deploy_batch40.prototxt'
cnn_model = '/home/ksharan1/visualization/san-vqa-tensorflow/vgg19/VGG_ILSVRC_19_layers.caffemodel'
vqa_model = '/home/ksharan1/visualization/san-vqa-tensorflow/model/san_lstm_att3/model-70000'
new_checkpoint= '/home/ksharan1/visualization/san-vqa-tensorflow/model/san_lstm_att3/model2-70000'

input_embedding_size = 512              # the encoding size of each token in the vocabulary
rnn_size = 256                          # size of the rnn in number of hidden nodes in each layer
rnn_layer = 2                           # number of the rnn layer
dim_image = (7, 7, 512)
dim_hidden = 1024                       # size of the common embedding vector
dim_attention = 512                   # size of attention embedding
num_output = 1000                       # number of output answers
max_words_q = 26
[IMG_HEIGHT, IMG_WIDTH] = [224, 224]
itoa = json.load(open(itoa_json, 'r'))
wtoi = json.load(open(wtoi_json, 'r'))
vocabulary_size = len(wtoi.keys())
##########################################################

def prepro_text(ques):

    # tokenize question
    txt = word_tokenize(ques.lower())
    question = [w if wtoi.get(w, len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
    print(question)

    # encode question (right_align)
    label_arrays = np.zeros((1, max_words_q), dtype='uint32')
    label_length = min(max_words_q, len(question)) # record the length of this sequence
    for i in range(2, label_length+1):
        label_arrays[0, -i] = wtoi[question[-i]]-1
    return label_arrays

def extract_imfeat(img):

    #base_model = VGG19(weights='imagenet')
    #model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    #model.summary()
    #img_path = 'train/dogs/1.jpg'
    #img = image.load_img(img_path, target_size=(224, 224))
    #mean = np.array((103.939, 116.779, 123.680), dtype=np.float32)
    #img_transform = misc.imresize(img, (IMG_HEIGHT, IMG_WIDTH)) - mean
    # img_data = image.load_img(img, target_size=(224, 224) )
    #img_data = np.expand_dims(img_transform, axis=0)
    #img_data = preprocess_input(img_data)
    #imfeat = model.predict(img_data)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(cnn_proto, cnn_model, caffe.TEST)
    mean = np.array((103.939, 116.779, 123.680), dtype=np.float32)
    I = misc.imresize(img, (IMG_HEIGHT, IMG_WIDTH))-mean
    I = np.transpose(I, (2, 0, 1))
    net.blobs['data'].data[:] = np.array([I])
    net.forward()
    imfeat = net.blobs['pool5'].data[...].copy()
    mag = np.sqrt(np.sum(np.multiply(imfeat, imfeat), axis=1))
    imfeat = np.transpose(imfeat,(0,2,3,1))
    imfeat = np.divide(imfeat, np.transpose(np.tile(mag,[512,1,1,1]),(1,2,3,0)) + 1e-8)
    return imfeat

def attention(img, att_map):
    att_map = np.reshape(att_map, [7,7])
    att_map = att_map.repeat(32, axis=0).repeat(32, axis=1)
    att_map = np.tile(np.expand_dims(att_map, 2),[1,1,3])
    att_map[:,:,1:] = 0
    # apply gaussian
    att_map = gaussian_filter(att_map, sigma=7)
    att_map = (att_map-att_map.min()) / att_map.max()
    att_map = cv2.resize(att_map, (img.shape[1], img.shape[0]))
    new_img = att_map*255*0.8 + img*0.2
    return new_img

def test(model_path='model/san_lstm_att3/model-70000'):

    print("Model Path: " + model_path)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #tf.initialize_all_variables().run(session=sess)
    #with tf.Session() as sess:
    #    tf.initialize_all_variables()
    model = Answer_Generator(
            rnn_size = rnn_size,
            rnn_layer = rnn_layer,
            batch_size = 1,
            input_embedding_size = input_embedding_size,
            dim_image = dim_image,
            dim_hidden = dim_hidden,
            dim_attention = dim_attention,
            max_words_q = max_words_q,
            vocabulary_size = vocabulary_size,
            drop_out_rate = 0,
            sess = sess)

    model.build_generator()

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    #new_checkpoint_vars = {}
    #reader = tf.train.NewCheckpointReader(vqa_model)
    sess.run(tf.initialize_all_variables())
    t_vars = tf.trainable_variables()
    with tf.device('/cpu: 0'):
    	saver = tf.train.Saver(t_vars)
    	saver.restore(sess, '/home/ksharan1/visualization/san-vqa-tensorflow/model/san_lstm_att3/model-70000')

    while 1:

        if os.path.isfile(vqa_data_json):
            # get data
            vqa_data = json.load(open(vqa_data_json, 'r'))
            ques = vqa_data['question']
            print ques
            imname = vqa_data['image']
            #print vqa_data['image']
            print imname
            img = cv2.imread(imname).astype(np.float32)
            os.remove(vqa_data_json)


            question = prepro_text(ques)
            imfeat = extract_imfeat(img)

            generated_ans, prob_att1, prob_att2 = model.test(imfeat, question)
            generated_ans = np.argmax(generated_ans, axis=1)[0]
            att2 = attention(img, prob_att2)
            cv2.imwrite(att_image, att2.astype(np.uint8))

            ans = itoa[str(generated_ans+1)]
            print ans
            with open(answer_txt, 'w') as f:
                f.write(ans)
            print('Q: {}'.format(ques))
            print('A: {}'.format(ans))
        time.sleep(1)


if __name__ == '__main__':

        test(vqa_model)
