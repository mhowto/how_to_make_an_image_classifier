# coding: utf-8

import numpy as np
import tensorflow as tf
from PIL import Image

img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# In[3]:


from keras.preprocessing.image import ImageDataGenerator


# ## Small ConvNet

# ### Model architecture definition

# In[4]:


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
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    # Pooling #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu)

    # Pooling #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Dense Layer
    flatten = tf.contrib.layers.flatten(inputs=pool3)
    dense = tf.layers.dense(inputs=flatten, units=64, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2, activation=tf.nn.sigmoid)

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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

# In[5]:


# used to rescale the pixel values from [0, 255]
datagen = ImageDataGenerator(rescale=1. / 255)

# In[6]:


# automagically retrives images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

# In[7]:


# Create the Estimator
classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model2")

# In[8]:


# Set up a Loggin Hook
# tensors_to_log = {"probabilites": "softmax_tensor"}
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


# In[9]:


def next_wrapper(it):
    def next():
        features, lables = it.next()
        return {"x": features}, lables

    return next


# In[10]:


# Train the model
# classifier.train(input_fn=next_wrapper(train_generator), steps=20000, hooks=[logging_hook])

# ## Evaluationg on validation set

# In[26]:


seen = 0


def validate_next_wrapper(it):
    def next():
        global seen
        print("seen:", seen, "it.n", it.n)
        # raise tf.errors.OutOfRangeError(None, None, "Already emitted 1 epoch.")
        if seen > it.n:
            raise tf.errors.OutOfRangeError(None, None, "Already emitted 1 epoch.")
        features, labels = it.next()
        seen += len(features)
        print("seen changed to:", seen)
        return {"x": features}, labels

    return next


# In[28]:


eval_results = classifier.evaluate(input_fn=validate_next_wrapper(validation_generator))
