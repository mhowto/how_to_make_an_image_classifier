
# coding: utf-8

# # Try 1

# ## Data loading

# In[1]:


import numpy as np
import tensorflow as tf
from PIL import Image


# In[2]:


img_width, img_height = 150, 150
#img_width, img_height = 224, 224


train_data_dir = 'data/train'
validation_data_dir = 'data/validation'


# ## Small ConvNet

# ### Model architecture definition

# In[3]:


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    input_layer = tf.reshape(features["x"], [-1, img_width, img_height, 3])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)
    
    # Pooling #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[3,3],
        padding='same',
        activation=tf.nn.relu)
    
    # Pooling #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3,3],
        padding='same',
        activation=tf.nn.relu)
    
    # Pooling #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
    # Dense Layer
    flatten = tf.contrib.layers.flatten(inputs=pool3)
    dense = tf.layers.dense(inputs=flatten, units=64, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    predicitions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predicitions=predicitions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metris (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predicitions["classes"])        
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# ### Load Training and Test Data

# In[3]:


import os
def import_images_from_directory(direcotry, target_size=None, samples=None):
    images = []
    labels = []

    count = 0
    classes = {c: i for i, c in enumerate(os.listdir(direcotry))}
    for c, i in classes.items():
        if samples is not None and count >= samples:
            break
        p = os.path.join(direcotry, c)
        for f in os.listdir(p):
            if samples is not None and count >= samples:
                break
            im = Image.open(os.path.join(p, f))
            if target_size is not None:
                im = resize(im, target_size)
            images.append(np.array(im, dtype="float32")/255)
            im.close()
            labels.append(i)
            count += 1
    return np.array(images), np.array(labels, dtype="float32")

def resize(im, target_size):
    return im.resize(target_size, Image.ANTIALIAS)


# In[4]:


train_images, train_labels = import_images_from_directory(train_data_dir, (img_width, img_height))


eval_images, eval_labels = import_images_from_directory(validation_data_dir, (150, 150))

# # Using a pre-trained model

# In[18]:


import os, time

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = os.path.join('.', "vgg16.npy")
            vgg16_npy_path = path
            print(path)
        
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
    
    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")
        
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")
        
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, "pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, "pool4")
        
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, "pool5")
        '''
        self.fc6 = self.fc_layer(self.pool5, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        '''
        self.flatten = tf.contrib.layers.flatten(inputs=self.pool5)
        self.dense6 = tf.layers.dense(self.flatten, 256, activation=tf.nn.relu)
        self.keep_prob = tf.placeholder(tf.float32)
        self.dropout7 = tf.nn.dropout(self.dense6, self.keep_prob)
        
        self.y_conv = tf.layers.dense(self.dropout7, 2)        
        #self.prob = tf.nn.softmax(self.dense8, name="prob")
        
        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time))) 
        return self.y_conv, self.keep_prob
        
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding='SAME')
            
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            
            return tf.nn.relu(bias)
    
    def max_pool(slef, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
    
    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            return  tf.nn.bias_add(tf.matmul(x, weights), biases)
        
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
    
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")
    
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='biases')
    
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='weights')
        


# In[17]:


def next_batch_generator(data, batch_size):
    idx= 0
    size =len(data)
    while True:
        start_idx = idx
        end_idx = idx + batch_size
        idx = end_idx % size
        result = []
        if end_idx < size:
            result = data[start_idx:end_idx]
        else:
            result = data[start_idx:] + data[0:idx]
        yield result

g = next_batch_generator([i for i in range (20)], 7)
print(g.__next__())
print(g.__next__())
print(g.__next__())
print(g.__next__())


# In[30]:


images = tf.placeholder(tf.float32, [None, img_height,img_width,3])
labels = tf.placeholder(tf.float32, [None])
#feed_dict = {images: train_images, labels: train_labels}
vgg = Vgg16()
with tf.name_scope("content_vgg"):
    y_conv, keep_prob = vgg.build(images)

onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),depth=2)
with tf.name_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=onehot_labels,
        # labels=labels,
        logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)
# loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
with tf.name_scope('rmspo_optimizer'):
    train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(onehot_labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
'''
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(np.array([0] *50 + [1]*50).reshape(-1, 1))
'''

batch_size = 50
train_images_g = next_batch_generator(train_images, batch_size)
train_labels_g = next_batch_generator(train_labels, batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        x_batch = train_images_g.__next__()
        y_batch = train_labels_g.__next__()
        # y_batch = y_batch.reshape(-1, 1)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                images: x_batch, labels: y_batch, keep_prob: 1.0})
            print('step %d, traing accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={images: x_batch, labels: y_batch, keep_prob:0.5})
    
    print('test accuracy %g' % accuracy.eval(feed_dict={images:eval_images, labels:eval_labels, keep_prob: 1.0}))
