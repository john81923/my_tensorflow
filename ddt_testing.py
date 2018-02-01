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

    if( args.tester ): # this part has problem
        with open('synset.txt') as name:
            for line in name:
                cate_name = line.rstrip()
                cate_id =  get_common_object_id( cate_name + '_train')
                #bus sheep cow airplane 
                _mean_vec = mean_vector( cate_id ,sess, vgg, images)
                _convX = Cov_mat(  _mean_vec ,cate_id, sess , vgg, images)
                _w , _v = LA.eig( _convX )
                with open( cate_name + '_covx.pk', 'wb') as _fp:
                    pickle.dump( _v, _fp)
                with open( cate_name + '_mean.pk', 'wb') as _fm:
                    pickle.dump( _mean_vec, _fm)

                #car_mean_vec  = mean_vector( car_id ,sess, vgg, images)
                #car_convX = Cov_mat(  car_mean_vec ,car_id, sess , vgg, images)
                #car_w , car_v = LA.eig( car_convX )
                #with open( 'car_covx', 'wb') as car_fp:
                #    pickle.dump(car_v, car_fp)
                #with open( 'car_mean', 'wb') as car_fm:
                #    pickle.dump(car_mean_vec, car_fm)

                print 'mean & covx ' + cate_name
                print ''

    else: 
        print 'img testing'
        with open('synset.txt') as name:
            for line in name:
                cate_name = line.rstrip()
                cate_id =  get_common_object_id( cate_name + '_train')
                _img_gen = get_img( cate_id )
                print cate_name
                print 'id length ', len( cate_id )
                _img_data = np.reshape( _img_gen.next(),(1,224,224,3)).astype('float32')
                
                with open(  'saved_pk/' +cate_name + '_covx.pk', 'rb') as fp:
                    v = pickle.load( fp)
                    _v = v.T
                with open(  'saved_pk/' +cate_name + '_mean.pk', 'rb') as fm:
                    _mean_vec = pickle.load( fm)
                dcptr = sess.run(vgg.conv5_4, feed_dict={ images: _img_data }) 
                p1 = np.zeros( (dcptr.shape[1] , dcptr.shape[2] ))
                count = 0
                for i in range( dcptr.shape[1]):
                    for j in range( dcptr.shape[2]):
                        p1[i,j] = np.inner( _v[0], ( dcptr[0,i,j]-  _mean_vec ))
                        if p1[i,j]>0:
                            count = count +1
                #find max and min 
                print 'count ', count 
                maxv = np.amax( p1 )
                minv = np.amin( p1 )
                print maxv, " ",minv

                img_in = np.reshape( _img_data , (224,224,3))
                p1 = scipy.misc.imresize( p1, (224,224) )
                scipy.misc.imsave( './saved_DDT_img/'+ cate_name +'_imgDt.jpg', p1 )
                scipy.misc.imsave( './saved_DDT_img/'+ cate_name +'_imgOr.jpg', img_in )

def Cov_mat( mean_vec ,smp ,sess, vgg, images):
    covx = np.zeros( (512,512) ,)
    img = get_img( smp )
    for n in range( len(smp) ):
        img_in = np.reshape( img.next(),(1,224,224,3)).astype('float32')
        dcptr = sess.run(vgg.conv5_4, feed_dict={ images: img_in}) 
        for i in range( dcptr.shape[1]):
            for j in range( dcptr.shape[2] ):
                covx = covx + np.outer( (dcptr[0,i,j]- mean_vec) , (dcptr[0,i,j]- mean_vec)  )
    covx = covx / ( len(smp)* dcptr.shape[1]* dcptr.shape[2] )
    return covx

def mean_vector(  smp,sess,vgg ,images ):
    smp_len = len( smp )
    img = get_img( smp ) # generator
    sum_vec = np.zeros( 512 , dtype = 'float32')
    for n in range( smp_len ): #one photo at a time
        img_in = img.next().astype( 'float32' )
        img_in = np.reshape( img_in ,(1,224,224,3) )
        dcptr = sess.run( vgg.conv5_4, feed_dict={ images: img_in}) 
        for i in range( dcptr.shape[1] ):
            for j in range( dcptr.shape[2]):
                sum_vec = sum_vec + dcptr[0,i,j]
    mean_vec = sum_vec / (smp_len * dcptr.shape[1]* dcptr.shape[2])
    return mean_vec.astype( 'float32')

     
def str2bool(v):
    return v.lower() in("yes","true","t","1")
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str2bool, default=False )# training or eval
    parser.add_argument('--load', help='load model')             # ~/cls_154epoch.npy
    parser.add_argument('--tester', type=str2bool, default=False ) # go to tester part.
    parser.add_argument('--cate')
    args = parser.parse_args()
    _main(args.load, args.train)
