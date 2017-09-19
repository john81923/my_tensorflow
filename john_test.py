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
    eval_path =  '../data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
    anno_path = '../data/VOCdevkit/VOC2007/Annotations/'
    image_path = '../data/VOCdevkit/VOC2007/JPEGImages/'
    category = 'trainval'
    file_dir = os.path.join(ids_path, '{}.txt'.format(category))
    print file_dir
    with open(file_dir) as f:
        ids = []
        for line in f:
            ids.append(line[0:6])
    with open(eval_path) as ef:
        eval_ids = []
        for line in ef:
            eval_ids.append(line[0:6])
    #ids =np.asarray(ids, dtype='int')
    #np.random.shuffle(ids)
    data_num = len(ids)
    anno_dir_ = os.path.join(anno_path, '{}.xml'.format(ids[2]) )
    image_dir_ = os.path.join(image_path , '{}.jpg'.format(ids[2]))
    #
    # path to img and labels
    image_data_eval , label_vec_eval = img_n_label( image_dir_, anno_dir_ ) 
    image_data_eval = image_data_eval.reshape( (1,224,224,3) )
    #
    print 'data_num : ', data_num
    print len(DataFunc.classes) , DataFunc.classes
    # eval data and labels
    eval_dbatch = []
    eval_lbatch = []
    for i in range(100):
        anno_dir_eval = os.path.join(anno_path, '{}.xml'.format( eval_ids[ i]) )
        image_dir_eval = os.path.join(image_path, '{}.jpg'.format( eval_ids[ i]) )
        eval_data ,eval_label =  img_n_label ( image_dir_eval, anno_dir_eval )
        eval_dbatch.append(eval_data)
        eval_lbatch.append(eval_label)
    eval_dbatch = np.asarray(eval_dbatch, dtype = 'float' )
    eval_lbatch = np.asarray(eval_lbatch, dtype = 'float' )
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
    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)
    print  'var count : ', vgg.get_var_count()
    sess.run( tf.global_variables_initializer())
    # cost
    cost = tf.reduce_sum(( vgg.prob - true_out)**2 )
    # train
    train = tf.train.GradientDescentOptimizer( 0.0001).minimize( cost)
    
    # data and labels to batches for train  
    for i in range( 10): #data_num/batch_size ) 
        print 'epoch {}'.format(i)
        image_batch = []
        label_batch = []
        for j in range (batch_size):
            data_id = i*batch_size+j
            anno_dir_ = os.path.join(anno_path, '{}.xml'.format( ids[ data_id]) )
            image_dir_ = os.path.join(image_path, '{}.jpg'.format( ids[ data_id]) )
            image_data, label_vec = img_n_label ( image_dir_, anno_dir_ )
            image_batch.append(image_data)
            label_batch.append(label_vec)
        image_batch = np.asarray(image_batch, dtype = 'float' )
        label_batch = np.asarray(label_batch, dtype = 'float' )
        print image_batch.shape, label_batch.shape
        # run
        sess.run( train, feed_dict = { images: image_batch, true_out: label_batch, train_mode: True })
        # evaluate
        eval_model(sess)

def eval_model(sess):
    prob = sess.run(vgg.prob, feed_dict={ images: eval_dbatch, train_mode: False}) 
    #utils.print_prob( prob[0], './synset.txt' )
    probs =   probs_threshold(prob)
    y_ = tf.placeholder(tf.float32, [None, 20])
    correct_prediction = tf.equal( probs ,y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={images: eval_dbatch, y_: eval_lbatch }))

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

