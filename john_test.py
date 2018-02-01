import tensorflow as tf
import os
import sys
import argparse
#import scipy.misc as scipy
from scipy import stats
from skimage.transform import resize
import cv2
import numpy as np
import DataFunc 
import vgg19_trainable as vgg19
import utils
from random import shuffle
import random

# process from img.jpg to data_array , label.xml to label_vec
def img_n_label(image_dir , anno_dir ):
    image_data ,height ,width = DataFunc.process_image( image_dir) 
    image_data = resize( image_data, (224,224 ),mode='reflect' )
    image_data = image_data.reshape(( 224, 224, 3))
    labels = DataFunc.process_anno( anno_dir ) 
    label_vec = label_to_vec(labels)
    label_vec = label_vec.reshape((20))
    return  image_data, label_vec

def label_to_vec(labels):
    l_arr = np.zeros(20)
    for i in labels:
        if l_arr[int(i)] == 0:
            l_arr[ int(i) ] += 1  
    return l_arr

def get_img_ids( category ):
    # paths 
    ids_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/'
    file_dir = os.path.join(ids_path, '{}.txt'.format(category))
    print file_dir
    with open(file_dir) as f:
        ids = []
        for line in f:
            ids.append(line[0:6])
    return ids 

def get_batch( batch_size, rand_list, ids  ):
    image_batch = []
    label_batch = []
    anno_path = '../data/VOCdevkit/VOC2007/Annotations/'
    image_path = '../data/VOCdevkit/VOC2007/JPEGImages/'
    for j in range (batch_size):
        j_id = rand_list[ j ]
        anno_dir_ = os.path.join(anno_path, '{}.xml'.format( ids[ j_id ]) )
        image_dir_ = os.path.join(image_path, '{}.jpg'.format( ids[ int(j_id) ]) )
        image_data, label_vec = img_n_label ( image_dir_, anno_dir_ )
        image_batch.append(image_data)
        label_batch.append(label_vec)
    image_batch = np.asarray(image_batch, dtype = 'float' )
    label_batch = np.asarray(label_batch, dtype = 'float' ) 
    return image_batch, label_batch

def get_eval_batch( batch_size, batch_id , eval_ids ):
    eval_dbatch = []
    eval_lbatch = []
    anno_path = '../data/VOCdevkit/VOC2007/Annotations/'
    image_path = '../data/VOCdevkit/VOC2007/JPEGImages/'
    for i in range(batch_size):
        ids_count = batch_id *batch_size + i
        anno_dir_eval = os.path.join(anno_path, '{}.xml'.format( eval_ids[ ids_count ]) )
        image_dir_eval = os.path.join(image_path, '{}.jpg'.format( eval_ids[ ids_count ]) )
        eval_data ,eval_label =  img_n_label ( image_dir_eval, anno_dir_eval )
        eval_dbatch.append(eval_data)
        eval_lbatch.append(eval_label)
    eval_dbatch = np.asarray(eval_dbatch, dtype = 'float' )
    eval_lbatch = np.asarray(eval_lbatch, dtype = 'float' )
    return eval_dbatch, eval_lbatch

def eval_model(sess, vgg , images , train_mode , batch_size) :
    print 'start eval '
    # eval data and labels
    catg = 'val'
    eval_ids  = get_img_ids( catg)
    eval_num = len( eval_ids )/batch_size
    # 
    acc_accumlate = 0.
    for j in range ( eval_num ):
        eval_dbatch ,eval_lbatch = get_eval_batch( batch_size, j, eval_ids )
        # 
        prob = sess.run(vgg.prob, feed_dict={ images: eval_dbatch, train_mode: False}) 
        #conv5 shape 32,14,14,512
        #pool5 shape 32, 7, 7,512
        probs =  probs_threshold2(prob)
        #print 'probs ',probs
        #print 'label ',eval_lbatch
        p_ = tf.placeholder(tf.int32, [32])
        y_ = tf.placeholder(tf.float32, [32, 20])
        #correct_prediction = tf.equal( p_ ,y_)
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #acc_accumlate += sess.run( accuracy, feed_dict={ p_: probs , y_: eval_lbatch })
        topFiver = tf.nn.in_top_k( y_ ,p_ ,1 )
        acc = sess.run(topFiver, feed_dict = { y_:eval_lbatch, p_:probs })
        acc_accumlate += sum( acc )/ float(batch_size)    
    print acc_accumlate / float(eval_num)
    #vgg.save_npy(sess, './cls_model' )

def _main( load_model, istrain):
    # train session 
    batch_size = 32
    sess = tf.Session()
    # placeholders
    images = tf.placeholder( tf.float32, [batch_size, 224,224,3])
    true_out = tf.placeholder( tf.float32, [batch_size, 20])
    train_mode = tf.placeholder( tf.bool )
    # vgg19 model
    vgg = vgg19.Vgg19(load_model)
    vgg.build(images, train_mode)
    # cost
    cost = tf.reduce_sum(( vgg.prob - true_out)**2 )
    # train
    train = tf.train.GradientDescentOptimizer( 0.0001).minimize( cost)
    sess.run( tf.global_variables_initializer())
    # prepare data and label for training step
    epoch_size = 200
    if (args.tester == False):
        # get imags id
        catg = 'trainval'
        ids = get_img_ids( catg )
        data_num = len( ids )
        # data and labels to batches for train
        for epoch in range( epoch_size ):
            shuffle(ids)
            print 'epoch {}'.format( epoch)
            for i in range( data_num/batch_size ): # total batch number
                rand_list = []
                for rand in range (batch_size):
                    rand_list.append( random.randint( 0, data_num-1 ) )
                # get each batch data 
                image_batch, label_batch = get_batch( batch_size, rand_list, ids)    
                if args.train:
                    sess.run( train, feed_dict = { images: image_batch, true_out: label_batch, train_mode: True })
                if( (i+1) % 60 ==0):
                    eval_model(sess, vgg , images , train_mode , batch_size ) 
    else:
        print 'Tester ......' 




def probs_threshold(probs):
    a = 0
    b = np.zeros(20)
    for i in range(probs.shape[0]):
        idx = np.argmax( probs[i] )
        #ddb[idx] = 1
        probs[i] = idx
        #a += np.count_nonzero(probs[i])
    return probs

def probs_threshold2(probs):
    a = 0
    b = np.zeros(probs.shape[0])
    for i in range(probs.shape[0]):
        idx = np.argmax( probs[i] )
        #ddb[idx] = 1
        b[i] = idx
        #a += np.count_nonzero(probs[i])
    return b
     
def str2bool(v):
    return v.lower() in("yes","true","t","1")
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str2bool, default=False )# training or eval
    parser.add_argument('--load', help='load model')             # ~/cls_154epoch.npy
    parser.add_argument('--tester', type=str2bool, default=False ) # go to tester part.
    args = parser.parse_args()
    _main(args.load, args.train)
