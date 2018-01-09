'''
Source: Shervine Amidi @ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

'''

import numpy as np
from tools import threadsafe_generator

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 40, dim_y = 40, dim_z = 40, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle

  @threadsafe_generator
  def generate(self, labels, list_IDs, plane):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp, plane)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, list_IDs_temp, plane):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.zeros((self.batch_size, self.dim_z, self.dim_y, self.dim_x, 1))
      y = np.empty((self.batch_size), dtype = int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          temp = (normalize(np.load(ID))) # normalize and zero-center the images
          
          if plane == 'xz':
            temp = np.rot90(temp,1,(1,0))
          elif plane == 'yz':
            temp = np.rot90(temp,1,(2,0))

          if self.dim_z == 1:
            X[i,0,:temp.shape[1],:temp.shape[2],0] = temp[int(temp.shape[0]/2),:,:]
          else:
            X[i,:temp.shape[0],:temp.shape[1],:temp.shape[2],0] = temp

          # Store class
          y[i] = labels[ID]

      return X, sparsify(y)

def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 2 # Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN
    return image