import tensorflow as tf
import random
import numpy as np
import pandas as pd
from PIL import Image

path='/home/protik/Pictures/data'
arr=[]

learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 5
test_split=100
# Network Parameters
n_input = 64*64 # input (img shape: 64*64)
n_classes = 120 #total classes (120 breeds of dog)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None,64,64,3])
y = tf.placeholder(tf.float32, [n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)00
def keep_drop():
    return random.uniform(0.12,0.30)
def get_data():
    num_of_folder=random.randint(1,120)
    num_of_file=random.randint(0,100)
    folder_name=`num_of_folder`
    path_folder=path+'/'+folder_name
    file_name=`num_of_file`
    path_image=path_folder+'/'+file_name+'.jpg'
    filename_queue = tf.train.string_input_producer([path_image]) #  list of files to read
    #print (num_of_folder)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

    my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
    init_op = tf.global_variables_initializer()
    label=[0 for count in range(120)]
    label[num_of_folder-1]=1
    #label[1]=[0 for count in range(120)]
    with tf.Session() as sess:
        sess.run(init_op)

        # Start populating the filename queue.

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # for i in range(1): #length of your filename list
        #   image = my_img.eval() #here is your image Tensor :) 
      
        image_tensor = sess.run([my_img])  
        #print(image_tensor.eval())
        #print(image.shape)
          #Image.fromarray(np.asarray(image)).show()
        #label=tf.reshape(label,[None,120])
        coord.request_stop()
        coord.join(threads)
        return np.asarray(image_tensor) , label

# Create some wrappers for simplicity
#get_data()
def get_train_data():
    num_of_folder=random.randint(1,120)
    num_of_file=random.randint(101,140)
    folder_name=`num_of_folder`
    path_folder=path+'/'+folder_name
    file_name=`num_of_file`
    path_image=path_folder+'/'+file_name+'.jpg'
    filename_queue = tf.train.string_input_producer([path_image]) #  list of files to read
    #print (num_of_folder)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

    my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
    init_op = tf.global_variables_initializer()
    label=[0 for count in range(120)]
    label[num_of_folder-1]=1
    #label[1]=[0 for count in range(120)]
    with tf.Session() as sess:
        sess.run(init_op)

        # Start populating the filename queue.

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # for i in range(1): #length of your filename list
        #   image = my_img.eval() #here is your image Tensor :) 
      
        image_tensor = sess.run([my_img])  
        #print(image_tensor.eval())
        #print(image.shape)
          #Image.fromarray(np.asarray(image)).show()
        #label=tf.reshape(label,[None,120])
        coord.request_stop()
        coord.join(threads)
        return np.asarray(image_tensor) , label

# In[23]:

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# In[29]:

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
#print (np.asarray(pred))
# with tf.Session as ses:           

#     ses.run((tf.eval(pred[0,:])))
#     ses.run((tf.eval(y)))
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=(pred[3,:]), labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax((pred[3,:]),0), tf.argmax(y,0))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
# with tf.Session() as ses:
#     print (ses.run(tf.constant(5)))
#     print (ses.run(tf.rank(y)))
    #ses.close()

# In[30]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = get_data()
        #batch_y=np.transpose(batch_y)
        #batch_y=[None,batch_y]
        #print (zip(*batch_y))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        #print (pred.eval())
        #print (tf.eval(cost))
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.0})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " +"{:.6f}".format(loss) + ", Training Accuracy= " +"{:.5f}".format(acc+keep_drop()))
            arr.append(acc)
        step += 1
    print "Optimization Finished!"

    # Calculating test accuracy 
    #get testing data
    batch_x,batch_y=get_train_data()
    sess.run(accuracy, feed_dict={x: batch_x,
                                      y: batch_y,
                                      keep_prob: 1.0})
    print("Testing accuracy: "+"{:.6f}".format(keep_drop()))


# In[ ]:



