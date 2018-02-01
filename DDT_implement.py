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
from numpy import linalg as LA
import cPickle as pickle
import scipy.misc

# process from img.jpg to data_array , label.xml to label_vec
def img_n_label(image_dir  ):
    image_data ,height ,width = DataFunc.process_image( image_dir) 
    image_data = resize( image_data, (224,224 ),mode='reflect' )
    image_data = image_data.reshape(( 224, 224, 3))
    return  image_data

def get_img( object_list ): 
    image_path = '../data/VOCdevkit/VOC2007/JPEGImages/'
    for i in range (len(object_list)):
        image_dir_ = os.path.join(image_path, '{}.jpg'.format( object_list[ int(i) ]) )
        image_data  = img_n_label ( image_dir_ )
        img = np.asarray(image_data, dtype = 'float' )
        yield img

def get_common_object_id( category ):
    ids_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/'
    file_dir = os.path.join( ids_path, '{}.txt'.format(category))
    print file_dir
    with open(file_dir) as f:
        ids = []
        for line in f:
            smp =  line.split( )
            if( smp[1] == '1'):
                ids.append(smp[0])
    return ids 


def _main( load_model, istrain):
    # train session 
    batch_size = 1
    sess = tf.Session()
    # placeholders
    images = tf.placeholder( tf.float32, [batch_size, 224,224,3])
    train_mode = tf.placeholder( tf.bool )
    # vgg19 model
    vgg = vgg19.Vgg19(load_model)
    vgg.build(images, train_mode)
    # train
    sess.run( tf.global_variables_initializer())
    # A set of N imgaes containing the common object
    smp = get_common_object_id( 'bus_train' )
    img = get_img(smp)
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    img_in = np.reshape( img.next(),(1,224,224,3))
    
    if( args.tester ): # this part has problem
        print 'counting cov mat'
        mean_vec  = mean_vector( smp ,sess, vgg, images)
        convX = Cov_mat(  mean_vec ,smp, sess , vgg, images)
        w ,v = LA.eig( convX )
        with open( 'DDT_covx', 'wb') as fp:
            pickle.dump(v, fp)
        with open( 'DDT_mean', 'wb') as fm:
            pickle.dump(mean_vec, fm)
    else: 
        print 'output img'
        with open( 'DDT_covx', 'rb') as fp:
            v = pickle.load( fp)
        v = v.T
        with open( 'DDT_mean', 'rb') as fm:
            mean_vec = pickle.load( fm)
        dcptr = sess.run(vgg.conv5_4, feed_dict={ images: img_in}) 
        p1 = np.zeros( (dcptr.shape[1] , dcptr.shape[2] ))
        print ''
        print v[0]
        print ''
        print mean_vec
        for i in range(dcptr.shape[1]):
            for j in range(dcptr.shape[2]):
                p1[i,j] = np.inner( v[0], ( dcptr[0,i,j]- mean_vec ))
        #find max and min 
        maxv = np.amax( p1 )
        minv = np.amin( p1 )
        print maxv, " ",minv
        p1 = ( p1-minv )/ ( maxv - minv )
        #for i in range ( p1.shape[0]):
        #    for j in range( p1.shape[1]):
        #        if ( i > 0 ):
        #            img_in[0,i,j]= 200 
        #        else :
        #            img_in[0,i,j]= 0
        img_in = np.reshape( img_in, (224,224,3))
        p1 = scipy.misc.imresize( p1, (224,224) )
        scipy.misc.imsave( 'outfile.jpg', p1 )
        scipy.misc.imsave( 'outfile2.jpg', img_in )

def Cov_mat( mean_vec ,smp ,sess, vgg, images):
    covx = np.zeros( (512,512) )
    img = get_img( smp )
    for n in range( len(smp) ):
        img_in = np.reshape( img.next(),(1,224,224,3))
        dcptr = sess.run(vgg.conv5_4, feed_dict={ images: img_in}) 
        for i in range( dcptr.shape[1]):
            for j in range( dcptr.shape[2] ):
                covx = covx + np.outer( (dcptr[0,i,j]- mean_vec) , (dcptr[0,i,j]- mean_vec)  )
    covx = covx / ( len(smp)* dcptr.shape[1]* dcptr.shape[2] )

    return covx

def mean_vector(  smp,sess,vgg ,images ):
    smp_len = len( smp)
    img = get_img( smp) # generator
    mean_vec = np.zeros( 512)
    for i in range( smp_len ): #one photo at a time 
        img_in = np.reshape( img.next(),(1,224,224,3))
        dcptr = sess.run( vgg.conv5_4, feed_dict={ images: img_in}) 
        sum_vec = np.zeros( dcptr.shape[3])
        for i in range( dcptr.shape[1] ):
            for j in range( dcptr.shape[2]):
                sum_vec = sum_vec + dcptr[0,i,j]
        mean_ij = sum_vec / pow( dcptr.shape[1], 2)
        mean_vec = mean_vec + mean_ij 
    mean_vec = mean_vec/ smp_len
    return mean_vec

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
