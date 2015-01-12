#!/usr/bin/python
import sys
import datetime
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import matplotlib.pyplot as plt
import cPickle
try:
  import PIL.Image as Image
except ImportError:
  import Image
import pandas as pd
from utils import tile_raster_images
from load_images import load_train
from load_images import load_test

"""
Convolutional Neural Network
============================
In this case applied to the image dataset of the NDSB Plankton Image recognition on Kaggle.
"""

#srng = RandomStreams(seed=12469)
srng = RandomStreams()

def floatX(X):
  return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
  return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
  return T.maximum(X, 0.)

def softmax(X):
  e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
  return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
  if p > 0:
    retain_prob = 1 - p
    X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    X /= retain_prob
  return X

# default learning rate 0.001
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
  grads = T.grad(cost=cost, wrt=params)
  updates = []
  for p, g in zip(params, grads):
    acc = theano.shared(p.get_value() * 0.)
    acc_new = rho * acc + (1 - rho) * g ** 2
    gradient_scaling = T.sqrt(acc_new + epsilon)
    g = g / gradient_scaling
    updates.append((acc, acc_new))
    updates.append((p, p - lr * g))
  return updates

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
  """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
  predictions = np.clip(y_pred, eps, 1 - eps)

  # normalize row sums to 1
  predictions /= predictions.sum(axis=1)[:, np.newaxis]

  actual = np.zeros(y_pred.shape)
  n_samples = actual.shape[0]
  actual[np.arange(n_samples), y_true.astype(int)] = 1
  vectsum = np.sum(actual * np.log(predictions))
  loss = -1.0 / n_samples * vectsum
  return loss

def remap_for_image(weight_array):
  collength = len(weight_array)
  rowlength = len(weight_array[0])
  num_channels = 0
  num_kernels = 0
  try:
    P[0][0]
    num_channels = len(P[0][0])
    P[0][0][0]
    num_kernels = len(P[0][0][0])
  except (IndexError, TypeError):
    # do nothing
    pass
  if(num_kernels == 0):
    #hack for 2D arrays
    if(collength == 1024):
      t1 = []
      for i in range(rowlength):
        t2 = []
        for j in range(collength):
          t2.append(weight_array[j][i])
        npar2 = np.asarray(t2, dtype=theano.config.floatX)
        t1.append(npar2)
      npar1 = np.asarray(t1, dtype=theano.config.floatX)
      return npar1
    else:
      #print "no kernels: return original array"
      return weight_array

  remapped = []
  for i in range(num_kernels):
    for j in range(num_channels):
      img_vector = []
      for k in range(rowlength):
        for l in range(collength):
          img_vector.append(weight_array[l][k][j][i])
      nparr_s = np.asarray(img_vector, dtype=theano.config.floatX)
      remapped.append(nparr_s)
  M = np.asarray(remapped, dtype=theano.config.floatX)
  #print type(M)
  #print M.shape
  #print M.dtype
  return M

def save_plots(PY_hat, Y, num_plots, epoch):
  x = np.arange(len(PY_hat[0]))
  fig, axarr = plt.subplots(num_plots, 1, sharex=True, sharey=True)
  fig.set_size_inches(7.0,10.0)
  axarr[0].set_title('First ' + str(num_plots) + ' validation set predictions')
  for p in range(num_plots):
    axarr[p].plot(x, PY_hat[p], 'b-')
    axarr[p].tick_params(axis='both', which='major', labelsize=8)
    #axarr[p].tick_params(axis='both', which='minor', labelsize=6)
    axarr[p].set_xlabel('classes', fontsize=8)
    axarr[p].set_ylabel('probability', fontsize=8)
    axarr[p].axvline(Y[p], color='r', linestyle='solid')
    axarr[p].set_ylim(0.0, 1.0)
    
  fig.subplots_adjust(hspace=0.3)
  fig.savefig(('plots/plot_epoch_%03d.png' % epoch), dpi=80)
    
def save_weights():
  print "Saving model weights"
  save_file = open('saved_params/weights.pkl', 'wb')  # this will overwrite current contents
  cPickle.dump(w.get_value(borrow=True), save_file, -1)
  cPickle.dump(w2.get_value(borrow=True), save_file, -1)
  cPickle.dump(w22.get_value(borrow=True), save_file, -1)
  #cPickle.dump(w222.get_value(borrow=True), save_file, -1)
  cPickle.dump(w3.get_value(borrow=True), save_file, -1)
  cPickle.dump(w4.get_value(borrow=True), save_file, -1)
  cPickle.dump(w_o.get_value(borrow=True), save_file, -1)
  save_file.close()

def load_weights():
  print "Loading existing model weights"
  save_file = open('saved_params/weights.pkl')
  w.set_value(cPickle.load(save_file), borrow=True)
  w2.set_value(cPickle.load(save_file), borrow=True)
  w22.set_value(cPickle.load(save_file), borrow=True)
  #w222.set_value(cPickle.load(save_file), borrow=True)
  w3.set_value(cPickle.load(save_file), borrow=True)
  w4.set_value(cPickle.load(save_file), borrow=True)
  w_o.set_value(cPickle.load(save_file), borrow=True)
  save_file.close()

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
  l1a = rectify(conv2d(X, w, border_mode='full'))
  l1 = max_pool_2d(l1a, (2, 2))
  l1 = dropout(l1, p_drop_conv)

  l2a = rectify(conv2d(l1, w2))
  l2 = max_pool_2d(l2a, (2, 2))
  l2 = dropout(l2, p_drop_conv)

  l3a = rectify(conv2d(l2, w3))
  l3b = max_pool_2d(l3a, (2, 2))
  l3 = T.flatten(l3b, outdim=2)
  l3 = dropout(l3, p_drop_conv)

  l4 = rectify(T.dot(l3, w4))
  l4 = dropout(l4, p_drop_hidden)

  pyx = softmax(T.dot(l4, w_o))
  return l1, l2, l3, l4, pyx

# model2 = model but with added 3x3 convolutional layer
def model2(X, w, w2, w22, w3, w4, p_drop_conv, p_drop_hidden):
  l1a = rectify(conv2d(X, w, border_mode='full'))
  l1 = max_pool_2d(l1a, (2, 2))
  l1 = dropout(l1, p_drop_conv)

  l2a = rectify(conv2d(l1, w2))
  l2 = max_pool_2d(l2a, (2, 2))
  l2 = dropout(l2, p_drop_conv)

  l22a = rectify(conv2d(l2, w22))
  l22 = max_pool_2d(l22a, (2, 2))
  l22 = dropout(l22, p_drop_conv)


  l3a = rectify(conv2d(l22, w3))
  l3b = max_pool_2d(l3a, (2, 2))
  l3 = T.flatten(l3b, outdim=2)
  l3 = dropout(l3, p_drop_conv)

  l4 = rectify(T.dot(l3, w4))
  l4 = dropout(l4, p_drop_hidden)

  pyx = softmax(T.dot(l4, w_o))
  return l1, l2, l22, l3, l4, pyx

# model3 = model but with 5 3x3 convolutional layers
def model3(X, w, w2, w22, w222, w3, w4, p_drop_conv, p_drop_hidden):
  l1a = rectify(conv2d(X, w, border_mode='full'))
  l1 = max_pool_2d(l1a, (2, 2))
  l1 = dropout(l1, p_drop_conv)

  l2a = rectify(conv2d(l1, w2))
  l2 = max_pool_2d(l2a, (2, 2))
  l2 = dropout(l2, p_drop_conv)

  l22a = rectify(conv2d(l2, w22))
  l22 = max_pool_2d(l22a, (2, 2))
  l22 = dropout(l22, p_drop_conv)

  l222a = rectify(conv2d(l22, w222))
  l222 = max_pool_2d(l222a, (2, 2))
  l222 = dropout(l222, p_drop_conv)

  l3a = rectify(conv2d(l222, w3))
  l3b = max_pool_2d(l3a, (2, 2))
  l3 = T.flatten(l3b, outdim=2)
  l3 = dropout(l3, p_drop_conv)

  l4 = rectify(T.dot(l3, w4))
  l4 = dropout(l4, p_drop_hidden)

  pyx = softmax(T.dot(l4, w_o))
  return l1, l2, l22, l222, l3, l4, pyx

###################################
# End of Functions
###################################

print "-=: National Data Science Bowl - Plankton Classification :=-"
print "            - - Convolutional Neural Network - -"
print

bool_train_model=True
bool_run_predictions=True
load_saved_weights=True
epoch_start_at = 0
epoch_run_for = 120
var_learning_rate = 0.0004

print "Training Model: %r" % bool_train_model
print "Running Predictions: %r" % bool_run_predictions
print
if(bool_train_model):
  print "Running training model for %i epochs, starting at %i" % (epoch_run_for, epoch_start_at)
else:
  print "Note: training images will load, but no training will run. "
  print
  
trX, trY, trylabel, valX, valY, valylabel, class_names, num_features  = load_train()

print
print "Initializing...."

trX = trX.reshape(-1, 1, 96, 96)
valX = valX.reshape(-1, 1, 96, 96)

X = T.tensor4(theano.config.floatX)
Y = T.matrix(theano.config.floatX)

print "Initializing weight values..."
"""
Model 3 layers weights initialization

num_classes = len(class_names)
w = init_weights((32, 1, 5, 5))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((3200, 1024))
w_o = init_weights((1024, num_classes))
"""

"""
Model 4 conv layers weights initialization
"""
num_classes = len(class_names)
w = init_weights((32, 1, 5, 5))
w2 = init_weights((64, 32, 5, 5))
#for model2 and 3
w22 = init_weights((128, 64, 3, 3))
#for model 3
#w222 = init_weights((256, 128, 3, 3))
w3 = init_weights((256, 128, 3, 3))
w4 = init_weights((6400, 1024))
w_o = init_weights((1024, num_classes))

"""
12/22: 2304 hidden layer nodes gave best 1.922 on test set with 200 epochs - with 1.5 on val set around 70 epochs
overfit danger: testing lower number...
trying 2025 = 45**2  ==> gave lowest validation MCLL of 1.38 on epoch 39-40 (submitted validation MCLL 1.54 - Kaggle score 1.555
trying 1024: quit successful! lowest MCLL of ~1.35 around epoch 48 and possibly still improving...
on a hunch increased filter of layer 3 from 3x3 to 4x4, and in a trial run that proved quite interesting with a 1.31 validation MCLL
while on epoch ~44 of a 50 epoch run.
Sadly, only got to 1.41 on a longer run...

Amazing how reducing the hidden layer nodes is working. 484 is giving ~ 1.21-22 at best...!

model2:
trying to add another 3x3 convolution and sampling layer

Original:
num_classes = len(class_names)
w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, num_classes))

Note w_4 = 3200, something for our image size and 3 conv layers

1/7: tried various new things: larger image sizes... not-skewing the image by first making square by adding white space
5-layer 3x3 conv layers gave at best 1.13, but took like 12-20 hrs to run with 112x112 images. Adjustable learning rates
(and learning rates way lower than previous)... Also 75% drop out for hidden layer, 25% for conv layers

Now returning to 4 layers... but keeping larger image size...
"""


if(load_saved_weights):
  try:
    load_weights()
  except Exception, e:
    print "Error loading weights: %s" % e
    print "Initializing from random."
    pass



print "Building model..."
#noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
#l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)

noise_l1, noise_l2, noise_l22, noise_l3, noise_l4, noise_py_x = model2(X, w, w2, w22, w3, w4, 0.25, 0.75)
l1, l2, l22, l3, l4, py_x = model2(X, w, w2, w22, w3, w4, 0., 0.)

#noise_l1, noise_l2, noise_l22, noise_l22, noise_l3, noise_l4, noise_py_x = model3(X, w, w2, w22, w222, w3, w4, 0.25, 0.75)
#l1, l2, l22, l222, l3, l4, py_x = model3(X, w, w2, w22, w222, w3, w4, 0., 0.)

ty_x = T.argmax(noise_py_x, axis=1)
tpy_hat = noise_py_x

y_x = T.argmax(py_x, axis=1)
py_hat = py_x

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
#params = [w, w2, w22, w222, w3, w4, w_o]
params = [w, w2, w22, w3, w4, w_o]
#params = [w, w2, w3, w4, w_o]
#updates = RMSprop(cost, params, lr=0.001)
updates = RMSprop(cost, params, lr=var_learning_rate)

train = theano.function(inputs=[X, Y], outputs=[cost, ty_x, tpy_hat], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[y_x,py_hat], allow_input_downcast=True)
print "Initialization complete."


print

if(bool_train_model):
  print "Starting training"
  print "Learning rate: %f" % var_learning_rate
  """
  min_val_mcl_loss will be used to track how we're doing against the validation set in our predictions.
  The danger is that we will overfit the model and it does better and better against the training set, while not
  doing too well on the validation (and therefore - presumably - the test set. Some runs towards 200 epochs showed that
  the best validation MCLL score was ~1.5 while it ended at 1.92 (and the score on the test set via Kaggle matched that)
  So, track this to be able to show when we haven't improved for 10 epochs, and send a warning when we haven't for 5.
  Use this to determine whether you'd want to manually quit. The weights are saved after each epoch, so just turn off
  training after stopping and run the predictions by running the script again.
  4.795791 is the "Uniform Probability Benchmark ( 1/ number of classes ) Beat this to be better than uniformly random
  """
  min_val_mcl_loss = 4.795791  
  not_improved_in = 0
  #############################################################################################
  # run for epochs_run_for epochs. Change epoch_start_at up above to start later (for instance, after restart w/ weights)
  for i in range(epoch_start_at, epoch_start_at + epoch_run_for):
    """adjust variable learning rate if needed
    """
    if i==10:
      var_learning_rate = 0.0002
      print "Learning rate adjusted: %f" % var_learning_rate
    elif i==20:
      var_learning_rate = 0.0001
      print "Learning rate adjusted: %f" % var_learning_rate
    elif i==30:
      var_learning_rate = 0.00002
      print "Learning rate adjusted: %f" % var_learning_rate
      
    for start, end in zip(range(0, len(trX), 64), range(64, len(trX), 64)):
    #for start, end in zip(range(0, 1000, 128), range(128, 1000, 128)):
      cost, TY_hat, TPY_hat = train(trX[start:end], trY[start:end])
      print "Epoch %d: batch [%d-%d]: cross entropy cost = %f" % (i, start, end, cost)
      print "- Training Model Performance: \t%f" % (np.mean(np.argmax(trY[start:end], axis=1) == TY_hat))

      mcl_loss = multiclass_log_loss(np.argmax(trY[start:end], axis=1), TPY_hat)
      print "- Multi Class Log Loss: \t%f" % mcl_loss
      #print "- Delta Cat. cross entropy - Multi Class Log Loss: %f" % (cost - mcl_loss)

      print "First 15 Train Predictions (^) and Train Data (v):"
      print TY_hat[:15]
      sub_trY= trY[start:end]
      print np.argmax(sub_trY[:15], axis=1)

      print
    cost, TY_hat, TPY_hat = train(trX[end:len(trX)], trY[end:len(trY)])
    print "Epoch %d: batch [%d-%d]: cross entropy cost = %f" % (i, end, len(trY), cost)
    print "- Training Model Performance: \t%f" % (np.mean(np.argmax(trY[end:len(trY)], axis=1) == TY_hat))

    mcl_loss = multiclass_log_loss(np.argmax(trY[end:len(trY)], axis=1), TPY_hat)
    print "- Multi Class Log Loss: \t%f" % mcl_loss
    #print "- Delta Cat. cross entropy - Multi Class Log Loss: %f" % (cost - mcl_loss)

    print "First 15 Train Predictions (^) and Train Data (v):"
    print TY_hat[:15]
    sub_trY= trY[end:len(trY)]
    print np.argmax(sub_trY[:15], axis=1)
    
    print
    #print params
    #print
    count=0
    for p in params:
      count += 1
      print "Layer %i weights" % count
      P = p.get_value(borrow=True).T
      
      print P.shape
      img_arr = remap_for_image(P)
      numtiles = len(img_arr)
      tilecols = 64
      if(numtiles < 128):
        tilecols = 10
      tilerows = len(img_arr)/tilecols + 1
      size_img = np.sqrt(len(img_arr[0]))

      image = Image.fromarray(
        tile_raster_images(
          X=img_arr,
          img_shape=(size_img,size_img),
          tile_shape=(tilerows, tilecols),
          tile_spacing=(1,1)
        )
      )
      print "Saving filter image L%i..." % count
      image.save('filters/filters_l%i_at_epoch_%03d.png' % (count,i))
      print

    
    print "Running prediction over validation set"
    """
    Note that this doesn't check for size. The validation prediction run with 12,000 images takes
    around 4GB of memory on a GTC TITAN using 48x48 sized images and these model parameters/node sizes.
    If you run out of memory, you'll have to batch it, like I did with the test prediction run.

    With the full 130,400 test set, the predict function throws OoM errors
    """
    #Y_hat, PY_hat = predict(valX)

    #have to implement batching now for validation test. Running OoM
    val_Y_hat = np.zeros((len(valX), len(class_names)), dtype=float)
    """Watch out with the batch_size: on a GTX TITAN running the validation prediction of
    some 12,000 images took around 4GB peak of memory, using these model parameters.
    Even the train model run of 256 images takes ~700MB peak.
    With the full 130,400 test set, the predict function throws OoM errors (hence the batching)
    You may need to tune the batch_size for your GPU (if that's what you have)
    """
    
    batch_size = 2000
    print "Starting validation of %i images" % len(valX)
    print "Batches of %i images. Trying to do valX all in one go OoM's the GPU" % batch_size
    print "%i batches" % ((len(valX)/batch_size)+1)
    for start, end in zip(range(0, len(valX), batch_size), range(batch_size, len(valX), batch_size)):
      Y_hat, PY_hat = predict(valX[start:end])
      val_Y_hat[start:end] = PY_hat[0:(end-start)]
    #run the last batch, whatever the step function didn't catch
    Y_hat, PY_hat = predict(valX[end:len(valX)])
    val_Y_hat[end:len(valX)] = PY_hat
    
    
    print "Model prediction:"
    print np.mean(np.argmax(valY, axis=1) == np.argmax(val_Y_hat, axis=1))
    mcl_loss = multiclass_log_loss(np.argmax(valY, axis=1), val_Y_hat)
    print "- Multi Class Log Loss: \t%f" % mcl_loss

    """
    This handles the messaging around the Multi Class Log Loss over the validation set. It will show when it
    improves, but more importantly, it shows when it no longer does, and will show percentage delta.
    It would be a good idea to stop training when this loss score goes up, as it may indicate overfitting the
    training data
    """
    if(mcl_loss < min_val_mcl_loss):
      print "-----------------------------------------------------------"
      print "Validation MCL loss improvement from previous:\t%.2f%%" % (((min_val_mcl_loss/mcl_loss) * 100.0)-100.0)
      print "Current validation MCL loss:\t\t%f" % mcl_loss
      print "Previous minimum validation MCL loss:\t%f" % min_val_mcl_loss
      print "Improvement on last minimum since %i epochs" % not_improved_in
      print "-----------------------------------------------------------"
      min_val_mcl_loss = mcl_loss
      not_improved_in = 0                                         
    else:
      not_improved_in += 1
      print "No improvement on last minimum validation MCL loss in %i epochs." % not_improved_in
      print "Percentage MCL loss over min val MCL loss:\t%.2f%%" % ((mcl_loss/min_val_mcl_loss) * 100.0)
      print "Minimum validation MCL loss this run:\t\t%f" % min_val_mcl_loss
      if(not_improved_in == 5):
        print "--------------------------------------------"
        print "WARNING: No MCL loss improvement in 5 epochs"
        print "--------------------------------------------"
      elif(not_improved_in >= 10):
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        print "____________________________________________________"
        print "NOTICE: No MCL loss improvement in 10 or more epochs"
        print "____________________________________________________"
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        
        
    print "First 15 Predictions (^) and Actual (v):"
    print np.argmax(val_Y_hat[:15], axis=1)
    print np.argmax(valY[:15], axis=1)
    save_weights()

    print "Saving plots epoch %i..." % i
    save_plots(val_Y_hat, np.argmax(valY, axis=1), 12, i)
    print "--------------------------------------------------------"
    print
    print

    if(mcl_loss < 1.0):
      print "MCLL < 1.0. Stopping training...."
      break

if(bool_run_predictions):
  print "Running Predictions"
  #labels = map(lambda s: s.split('/')[-1], class_names)
  testX, images = load_test(num_features)
  testX = testX.reshape(-1, 1, 96, 96)
  full_Y_hat = np.zeros((len(testX), len(class_names)), dtype=float)
  """Watch out with the batch_size: on a GTX TITAN running the validation prediction of
    some 12,000 images took around 4GB peak of memory, using these model parameters.
    Even the train model run of 256 images takes ~700MB peak.
    With the full 130,400 test set, the predict function throws OoM errors (hence the batching)
    You may need to tune the batch_size for your GPU (if that's what you have)
  """
  batch_size = 2000
  print "Starting predictions of %i test images" % len(testX)
  print "Batches of %i images. Trying to do testX all in one go OoM's the GPU" % batch_size
  print "%i batches" % (len(testX)/batch_size)
  for start, end in zip(range(0, len(testX), batch_size), range(batch_size, len(testX), batch_size)):
    Y_hat, PY_hat = predict(testX[start:end])
    full_Y_hat[start:end] = PY_hat[0:(end-start)]
    sys.stdout.write('.')
    sys.stdout.flush()
  #run the last batch, whatever the step function didn't catch
  Y_hat, PY_hat = predict(testX[end:len(testX)])
  full_Y_hat[end:len(testX)] = PY_hat
  sys.stdout.write('.')
  sys.stdout.flush()
  print
  print "Done predictions"
  #print PY_hat.shape
  #print PY_hat[-1:] 
  #print full_Y_hat.shape
  #print full_Y_hat[-1:]

  header = "acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified".split(',')

  df = pd.DataFrame(full_Y_hat, columns=class_names, index=images)
  df.index.name = 'image'
  df = df[header]
  print "Writing output CSV file"
  out_file = "out/submission_" + datetime.datetime.now().strftime('%Y%m%d_%Hh%Mm%S') + ".csv"
  df.to_csv(out_file)
  print "Done. Output file: %s" % out_file
  print "If you're happy with the results, set bool_train_model to False and save this file."
  
  print "Script completed. Exiting."

  


