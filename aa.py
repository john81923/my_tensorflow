import tensorflow as tf
import numpy as np
from sklearn import metrics


sess = tf.Session()
def probs_threshold(probs):
    a = 0
    b = np.zeros(probs.shape[0])
    for i in range(probs.shape[0]):
        idx = np.argmax( probs[i] )
        #ddb[idx] = 1
        b[i] = idx
        #a += np.count_nonzero(probs[i])
    return b

input_ = np.random.rand(2,5)
print input_
pre_ =  probs_threshold(input_)
print pre_


probs = [4,1 ]
eval_y = [ [0,0,0,1,1] , [1,0,1,0,0] ]
print eval_y
p_ = tf.placeholder(tf.float32, [2, 5])
y_ = tf.placeholder(tf.float32, [2, 5])

topFiver = tf.nn.in_top_k(eval_y ,pre_ ,1 )

#correct_prediction = tf.equal( p_ ,y_)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
bool_ =  sess.run( topFiver)
print sum(bool_)/2.

yy = [ [0,0,0,1,1  ],[1,0,0,0,0] ]

acc, acc_op = tf.metrics.accuracy(labels=y_, predictions=p_)
acc , acc_op = tf.metrics.mean_per_class_accuracy(y_,p_ ,5  )
sess.run( tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#print sess.run([ acc, acc_op], feed_dict ={ y_ :eval_y , p_:yy   }  )
#print(metrics.accuracy_score(yy, eval_y))
