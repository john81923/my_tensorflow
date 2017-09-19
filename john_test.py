import tensorflow as tf
import os
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

def _main():
    # paths 
    ids_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/'
    anno_path = '../data/VOCdevkit/VOC2007/Annotations/'
    image_path = '../data/VOCdevkit/VOC2007/JPEGImages/'
    category = 'trainval'
    file_dir = os.path.join(ids_path, '{}.txt'.format(category))
    print file_dir
    with open(file_dir) as f:
        ids = []
        for line in f:
            ids.append(line[0:6])
    #ids =np.asarray(ids, dtype='int')
    #np.random.shuffle(ids)
    data_num = len(ids)
    #
    # path to img and labels
    #
    print 'data_num : ', data_num
    print len(DataFunc.classes) , DataFunc.classes
    # train session 
    batch_size = 32
    sess = tf.Session()
    # 
    # placeholders
    images = tf.placeholder( tf.float32, [batch_size, 224,224,3])
    true_out = tf.placeholder( tf.float32, [batch_size, 20])
    train_mode = tf.placeholder( tf.bool )
    #
    # vgg19 model
    vgg = vgg19.Vgg19('/home/master/05/john81923/vgg19.npy')
    vgg.build(images, train_mode)
    print  'var count : ', vgg.get_var_count()
    sess.run( tf.global_variables_initializer())
    # cost
    cost = tf.reduce_sum(( vgg.prob - true_out)**2 )
    # train
    train = tf.train.GradientDescentOptimizer( 0.0001).minimize( cost)
    # data and labels to batches for train
    print 'epoch_total = ', data_num / batch_size 
    for epoch in range(200):
        print 'epoch {}'.format(epoch)
        shuffle(ids)
        for i in range( data_num/batch_size ): #data_num/batch_size )
            #print "{} in range of {}dd".format(i,data_num/batch_size)
            image_batch = []
            label_batch = []
            rand_list = []
            for rand in range (batch_size):
               rand_list.append( random.randint( 0, data_num ) )
            for j in range (batch_size):
                j_id = rand_list[ j ]
                anno_dir_ = os.path.join(anno_path, '{}.xml'.format( ids[ j_id ]) )
                image_dir_ = os.path.join(image_path, '{}.jpg'.format( ids[ int(j_id) ]) )
                image_data, label_vec = img_n_label ( image_dir_, anno_dir_ )
                image_batch.append(image_data)
                label_batch.append(label_vec)
            image_batch = np.asarray(image_batch, dtype = 'float' )
            label_batch = np.asarray(label_batch, dtype = 'float' )
            # run
            #sess.run( train, feed_dict = { images: image_batch, true_out: label_batch, train_mode: True })
        # evaluate
        eval_model(sess ,vgg , images, train_mode)

def eval_model(sess, vgg , images , train_mode):
    eval_path =  '../data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
    anno_path = '../data/VOCdevkit/VOC2007/Annotations/'
    image_path = '../data/VOCdevkit/VOC2007/JPEGImages/'
    # eval data and labels
    with open(eval_path) as ef:
        eval_ids = []
        for line in ef:
            eval_ids.append(line[0:6])
    batch_size = 32 
    acc_accumlate = 0.
    eval_num = len( eval_ids )/batch_size
    for j in range ( eval_num ):
        eval_dbatch = []
        eval_lbatch = []
        for i in range(batch_size):
            ids_count = j*batch_size + i
            anno_dir_eval = os.path.join(anno_path, '{}.xml'.format( eval_ids[ ids_count ]) )
            image_dir_eval = os.path.join(image_path, '{}.jpg'.format( eval_ids[ ids_count ]) )
            eval_data ,eval_label =  img_n_label ( image_dir_eval, anno_dir_eval )
            eval_dbatch.append(eval_data)
            eval_lbatch.append(eval_label)
        eval_dbatch = np.asarray(eval_dbatch, dtype = 'float' )
        eval_lbatch = np.asarray(eval_lbatch, dtype = 'float' )
        prob = sess.run(vgg.prob, feed_dict={ images: eval_dbatch, train_mode: False}) 
        probs =  probs_threshold(prob)
        p_ = tf.placeholder(tf.float32, [None, 20])
        y_ = tf.placeholder(tf.float32, [None, 20])
        correct_prediction = tf.equal( p_ ,y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_accumlate += sess.run( accuracy, feed_dict={ p_: probs , y_: eval_lbatch })
    print acc_accumlate / eval_num
    vgg.save_npy( sess, './synset.txt' )

def probs_threshold(probs):
    a = 0
    b = np.zeros(20)
    for i in range(probs.shape[0]):
        #probs[i] = stats.threshold(probs[i], threshmin=0.1, newval=0)
        #probs[i] = stats.threshold(probs[i], threshmax=0.1, newval=1)
        idx = np.argmax( probs[i] )
        b[idx] = 1
        probs[i] = b
        a += np.count_nonzero(probs[i])
    return probs

if __name__ == '__main__':
    _main()

