import numpy as np
import random
from osgeo import gdal
import cv2
import tensorflow as tf
import cv2


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('zooming', 2.5, 'Amount to rescale original image.')
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate for ADAM solver.')
flags.DEFINE_integer('ws', 32, 'Window size for image clip.')
flags.DEFINE_integer('filter_size', 9, 'Filter size for convolution layer.')
flags.DEFINE_integer('filters', 32, 'Number of filters in convolution layer.')
flags.DEFINE_integer('batch_size', 1, 'Mini-batch size.')
flags.DEFINE_integer('max_runup_steps', 25, 'Maximum number of steps with deconvolution tied to convolution.')
flags.DEFINE_integer('max_layer_steps', 25, 'Maximum number of steps per layer.')
flags.DEFINE_integer('total_layers', 5, 'Number of perturbative layers in generative network.')
flags.DEFINE_integer('gpu',0,'GPU index.')
flags.DEFINE_bool('save',True,'Save output network')
flags.DEFINE_integer('numberOfBands',3,'Number of bands in training image.')
flags.DEFINE_float('delta',2.0,' Minimum dB improvement for new layer measured in PSNR.')
flags.DEFINE_float('min_alpha',0.0001,'Minimum value of perturbative layer parameter.')
flags.DEFINE_integer('layers_trained',0,'Number of layers previously trained.')
flags.DEFINE_string('training_image','../../datasets/Rio/3-Band/013022223103.tif','Training image.')
flags.DEFINE_integer('precision',8,'Number of bits per bands.')
flags.DEFINE_integer('convolutions_per_layer',2,'Number of convolutional layers in each perturbative layer.')


cpu = "/cpu:0"
gpu = "/gpu:"+str(FLAGS.gpu)
prefix = str(FLAGS.numberOfBands)+'-band'
numberOfBands = FLAGS.numberOfBands
maxp = 2.0**(1.0*FLAGS.precision) - 1.0


# Define the path for saving the neural network
summary_name = prefix + "." +str(FLAGS.total_layers) +"."
layer_name = '../../checkpoints/'+prefix+'/'+summary_name + '_' + str(FLAGS.zooming) + '_' + str(FLAGS.ws)










# Define CosmiQNet
# since the number of layers is determined at runtime, we initialize variables as arrays
sr = range(FLAGS.total_layers)
sr_cost = range(FLAGS.total_layers)
optimizer_layer = range(FLAGS.total_layers)
optimizer_all = range(FLAGS.total_layers)
W = range(FLAGS.total_layers)
Wo = range(FLAGS.total_layers)
b = range(FLAGS.total_layers)
bo = range(FLAGS.total_layers)
conv = range(FLAGS.total_layers)
inlayer = range(FLAGS.total_layers)
outlayer = range(FLAGS.total_layers)
alpha = range(FLAGS.total_layers)
beta = range(FLAGS.total_layers)
for i in range(FLAGS.total_layers):
    W[i] = range(FLAGS.convolutions_per_layer)
    b[i] = range(FLAGS.convolutions_per_layer)
    conv[i] = range(FLAGS.convolutions_per_layer)
deconv = range(FLAGS.total_layers)
MSE_sr = range(FLAGS.total_layers)
PSNR_sr = range(FLAGS.total_layers)



# The NN
with tf.device(gpu):
    # Input is has numberOfBands for the pre-processed image and numberOfBands for the original image
    x = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, numberOfBands])
    

    for i in range(FLAGS.total_layers):
        with tf.name_scope("layer"+str(i)) as scope:
            # alpha and beta are pertubation layer bypass parameters that determine a convex combination of a input layer and output layer
            alpha[i] = tf.Variable(0.1, name='alpha_'+str(i))
            beta[i] = tf.maximum( FLAGS.min_alpha , tf.minimum ( 1.0 , alpha[i] ), name='beta_'+str(i))
            if (0 == i) :
                inlayer[i] = x
            else :
                inlayer[i] = outlayer[i-1]
            # we build a list of variables to optimize per layer
            vars_layer = [alpha[i]]            
 
            # Convolutional layers
            W[i][0] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,numberOfBands,FLAGS.filters], stddev=0.1), name='W'+str(i)+'.'+str(0))
            b[i][0] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='b'+str(i)+'.'+str(0))
            conv[i][0] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d( inlayer[i], W[i][0], strides=[1,1,1,1], padding='SAME'), b[i][0], name='conv'+str(i)+'.'+str(0))) 
            for j in range(1,FLAGS.convolutions_per_layer):
                W[i][j] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,FLAGS.filters,FLAGS.filters], stddev=0.1), name='W'+str(i)+'.'+str(j))  
                b[i][j] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='b'+str(i)+'.'+str(j))
                vars_layer = vars_layer + [W[i][j],b[i][j]]
                conv[i][j] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d( conv[i][j-1], W[i][j], strides=[1,1,1,1], padding='SAME'), b[i][j], name='conv'+str(i)+'.'+str(j))) 
 
            # Deconvolutional layer
            Wo[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,numberOfBands,FLAGS.filters], stddev=0.1), name='Wo'+str(i))
            bo[i] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='bo'+str(i))
            deconv[i] = tf.nn.relu( 
                            tf.nn.conv2d_transpose( 
                                tf.nn.bias_add( conv[i][FLAGS.convolutions_per_layer-1], bo[i]), Wo[i], [FLAGS.batch_size,FLAGS.ws,FLAGS.ws,numberOfBands] ,strides=[1,1,1,1], padding='SAME'))
            vars_layer = vars_layer + [Wo[i],bo[i]]

            # Convex combination of input and output layer
            outlayer[i] = tf.nn.relu( tf.add(  tf.scalar_mul( beta[i] , deconv[i]), tf.scalar_mul(1.0-beta[i], inlayer[i])))


            # sr is the super-resolution process.  It really only has enhancement meaning during the current layer of training.
            sr[i] = tf.slice(outlayer[i],[0,0,0,0],[-1,-1,-1,numberOfBands])
           
ginit = tf.global_variables_initializer()
saver = tf.train.Saver()



with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    ckpt = tf.train.get_checkpoint_state('../../checkpoints/3-band/')
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
    
    img = cv2.imread('test.jpeg')
    out_img = np.zeros((256,256,3))
    for i in range(0,256,32):
        for j in range(0,256,32):
            predictions = sess.run(sr[4], feed_dict={x: [img[i:i+32,j:j+32]]})
            out_img[i:i+32,j:j+32] = predictions[0]

    cv2.imwrite('test_correct.jpg', out_img)