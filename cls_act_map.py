import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imsave
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_class_map(label, conv, im_width):
    print conv.shape
    output_channels = int(conv.get_shape()[-1])
    conv_resized = tf.image.resize_bilinear(conv, [im_width, im_width])
    print conv_resized.shape
    with tf.variable_scope('GAP', reuse=True):
        label_w = tf.gather(tf.transpose(tf.get_variable('W')), label)
        print label_w.shape
        label_w = tf.reshape(label_w, [-1, output_channels, 1])
    conv_resized = tf.reshape(conv_resized, [-1, im_width * im_width, output_channels])
    classmap = tf.matmul(conv_resized, label_w)
    classmap = tf.reshape(classmap, [-1, im_width, im_width])
    return classmap


def inspect_class_activation_map(sess, class_activation_map, top_conv,
                                         images_test, labels_test, global_step, num_images, x, y_, y):
    
    output_dir = './map_img/'
    mkdir_p(output_dir)
    imsave('./map_img/image_test.png')
    conv6_val, output_val = sess.run([vgg.act_map , vgg.prob], feed_dict={x: imges_test})
    classmap_answer = sess.run(class_activation_map, feed_dict={y_: label_test, top_conv: conv6_val})
    
    plt.imshow(1)
    plt.imshow(classmap_answer , cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
    cmap_file = './cmap_.png'
    plt.savefig(cmap_file)
    plt.close()
