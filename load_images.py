#!/usr/bin/python

from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
import numpy as np
try:
  import PIL.Image as Image
except ImportError:
  import Image
from utils import tile_raster_images

import warnings
warnings.filterwarnings("ignore")

"""
make a copy of the image with the max of width and height, and fill with white
this way we get a square copy of the image prior to resize and avoid skewing
"""
def make_square(img):
  height, width = img.shape
  max_side = max([height, width])

  copy = np.zeros(shape=(max_side, max_side), dtype=np.uint8)
  copy.fill(255)
  for i in range(height):
    for j in range(width):
      copy[i][j] = img[i][j]
  # increase contrast a bit with a gamma correction
  copy = exposure.adjust_gamma(copy, gamma=1.8)
  return copy

def randomize_vectors(a, b, a_num_features):
  rand_idx = np.arange(len(a))
  np.random.shuffle(rand_idx)
  #print rand_idx
  c = np.zeros((len(a), a_num_features), dtype=float)
  d = np.zeros((len(a)))
  for i in range(len(a)):
    c[i] = a[rand_idx[i]]
    d[i] = b[rand_idx[i]]
  return c, d

def one_hot(x,n):
  if type(x) == list:
    x = np.array(x)
  x = x.flatten()
  o_h = np.zeros((len(x),n))
  o_h[np.arange(len(x)),x] = 1
  return o_h

"""
Loads training image data (X), a one-hotted Y, the numeric labels (y), the actual classnames (class_names) and the image size/number of features in
the image vector (num_features). Since images are read from the folder in sequence, and therefore class by class, I randomized the order
of X and y in randomize_vectors(). This is so that in batches of 128 (for instance), I don't want to train first on a specific class, but
train all together.

NOTE: the classnames are NOT randomized. That is, read the correct class label for any y like this:
classname_for_y_i = class_names[int(y[i])]

The number of features are returned as a helper. It provides the image size (plus any possible additional features) in pixels (i.e. rows * cols).
As long as no particular features are added to the image, and the image is square, you can of course sqrt this number to get width/height.
"""
def load_train():
  print "-=: National Data Science Bowl - Plankton Classification :=-"
  print "                   - - Data Loader - -"
  # get the classnames from the directory structure
  directory_names = list(set(glob.glob(os.path.join("orig_data","train", "*"))).difference(set(glob.glob(os.path.join("orig_data","train","*.*")))))

  ### Grab images and rescale
  print "Loading training images...."

  # get total number of training images
  num_train_images = 0
  for folder in directory_names:
    for filename_dir in os.walk(folder):
      for filename in filename_dir[2]:
        #read image files (jpg)
        if filename[-4:] != ".jpg":
          continue
        num_train_images += 1

  print "Number of training images: %i" % num_train_images

  #rescale images to 25x25: may want to increase to 32x32 later for higher img resolution (at cost of more processing time)
  #we reshape to 25+3*25+3 = 28x28 to assist filters on edges

  #never mind: 28x28 proves to be pretty small and loses a lot of detail. Let's go 32+3x32+3

  #48 now. What can you do. some source images are as large as 60x80, but others are 48x30, for instance
  #this seems about the reasonable max. the rest will have to be model tweaks.
  # increase to 56x56
  # and 64x64
  # and 80x80
  # and 96X96
  max_pixels = 96
  image_size = max_pixels**2
  num_rows = num_train_images
  num_features = image_size

  # X: feature vector of one row per image
  X = np.zeros((num_rows, num_features), dtype=float)
  # y: numeric class label
  y = np.zeros((num_rows), dtype=np.uint8)

  files = []
  # create training data
  i = 0
  label = 0
  # list of string class names
  class_names = list()

  test_image = True

  # walk the class folders
  for folder in directory_names:
    #append class name for each class
    current_class = folder.split(os.path.sep)[-1]
    class_names.append(current_class)
    for filename_dir in os.walk(folder):
      test_image = True
      for filename in filename_dir[2]:
        if filename[-4:] != ".jpg":
          continue

        namefile_img = "{0}{1}{2}".format(filename_dir[0], os.path.sep, filename)
        img = imread(namefile_img, as_grey=True)
        img = make_square(img)
        files.append(namefile_img)
        img = resize(img, (max_pixels, max_pixels))

        #print test image to make sure we understand what we feed in
        if(test_image):
          image = Image.fromarray(img)
          image.save('test_img/test_image_' + filename[:-4] + '.tiff')
          test_image = False

        #Store rescaled image pixels
        X[i,0:image_size] = np.reshape(img, (1, image_size))

        #store class label
        y[i] = label
        i += 1
        # progress report
        report = [int((j+1)*num_rows/20.) for j in range(20)]
        if i in report: print np.ceil(i * 100.0 / num_rows), "% done"
    label += 1


  print "Loading images completed. Randomizing order..."
  X, y = randomize_vectors(X, y, num_features)
  print "Number of classes: %i" % len(class_names)
  for n in range(len(class_names)):
    print "[%i]\t%s" % (n, class_names[n])

  # recast as int. For some reason randomizing it makes them floats...
  y = y.astype(int)

  Y = one_hot(y, len(class_names))
  print "Prepared full data set."

  """
  print y[:50]
  for i in range(20):
    print class_names[int(y[i])]
    print X[i]
  """
  # split in train and validation sets
  print "Splitting full training data set into 4/5th training and 1/5th validation set..."
  fifth = len(X)/5
  trX = X[:-fifth]
  valX = X[-fifth:]
  trY = Y[:-fifth]
  valY = Y[-fifth:]
  tr_y = y[:-fifth]
  val_y = y[-fifth:]
  print "Training data set: %i images" % len(trX)
  print "Validation set:\t%i" % len(valX)
  print "Done."

  return trX, trY, tr_y, valX, valY, val_y, class_names, num_features

def load_test(num_features):
  print "Loading test images"
  image_width=int(np.sqrt(num_features))
  fnames = glob.glob(os.path.join("orig_data", "test", "*.jpg"))
  num_test_images = len(fnames)
  print "Number of test images: %i" % num_test_images
  testX = np.zeros((num_test_images, num_features), dtype=float)

  images = map(lambda file_name: file_name.split('/')[-1], fnames)

  i=0
  #progress report in loading
  report = [int((j+i)*num_test_images/20.) for j in range(20)]
  for file_name in fnames:
    image = imread(file_name, as_grey=True)
    image = make_square(image)
    image = resize(image, (image_width, image_width))

    testX[i,0:num_features] = np.reshape(image, (1, num_features))
    i += 1
    if i in report: print np.ceil(i*100.0/num_test_images), "% done"

  #print images
  print "Test data set: %i images" % num_test_images
  print "Done"
  print
  return testX, images
  
  
