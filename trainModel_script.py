import tools
import pickle

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed
# from keras.layers import Conv3D, MaxPooling3D

from keras import optimizers
from DataGenerator import DataGenerator
from keras.models import load_model
import keras.backend as K

from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)


# Filepaths ---------------------------------------------------------------------
path_neg_fold = 'D:\\Lung_Data\\extracted_volumes\\negatives'
path_pos_fold_train = 'D:\\Lung_Data\\extracted_volumes\\positives_augmented'
path_pos_fold_test = 'D:\\Lung_Data\\extracted_volumes\\positives_base'
path_history = 'D:\\lung_test_results\\metrics_temporary'

# Parameters ---------------------------------------------------------------------
params = {'dim_x': 40,
          'dim_y': 40,
          'dim_z': 40,
          'batch_size': 16,
          'shuffle': True}
params_valid = {'dim_x': params['dim_x'],
          'dim_y': params['dim_y'],
          'dim_z': params['dim_z'],
          'batch_size': 1,
          'shuffle': True}
plane = 'xy'

# Datasets ---------------------------------------------------------------------
subset_range = range(0,10)
for cv_counter in [(9,8),(8,7),(7,6),(6,5),(5,4),(4,3),(3,2),(2,1),(1,0),(0,9)]: # for 10-fold cross validation
  test_fold = cv_counter[0]
  valid_fold = cv_counter[1]
  partition,labels = tools.partition_data(subset_range, test_fold, valid_fold, path_neg_fold, path_pos_fold_train, path_pos_fold_test)

  # Generators ---------------------------------------------------------------------
  training_generator = DataGenerator(**params).generate(labels, partition['train'], plane)
  validation_generator = DataGenerator(**params_valid).generate(labels, partition['validation'], plane)


  # Design model ---------------------------------------------------------------------
  # Sequential CNN+LSTM:
  vision_model = Sequential()
  vision_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(params['dim_y'], params['dim_x'], 1)))
  vision_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  vision_model.add(MaxPooling2D((2, 2)))
  vision_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  vision_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  vision_model.add(MaxPooling2D((2, 2)))
  vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  vision_model.add(MaxPooling2D((2, 2)))
  vision_model.add(Flatten())
  video_input = Input(shape=(params['dim_z'], params['dim_y'], params['dim_x'], 1))
  encoded_frame_sequence = TimeDistributed(vision_model)(video_input)
  encoded_frame_sequence = Dropout(0.5)(encoded_frame_sequence)
  encoded_video = LSTM(512, return_sequences=False)(encoded_frame_sequence)
  output = Dense(2, activation='softmax')(encoded_video)
  model = Model(inputs=video_input, outputs=output)

  # # 3D CNN:
  # model = Sequential()
  # model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=(params['dim_z'], params['dim_y'], params['dim_x'], 1)))
  # model.add(MaxPooling3D((2, 2, 2)))
  # model.add(Dropout(0.5))
  # model.add(Flatten())
  # model.add(Dense(10, activation='relu'))
  # model.add(Dropout(0.5))
  # model.add(Dense(2, activation='softmax'))


  # Set training parameters/callbacks ----------------------------------------------------
  # to deal with unbalanced dataset
  weight_counter = []
  for ii in range(0,len(partition['train'])):
      weight_counter.append(labels[partition['train'][ii]])
  class_weight = {0:1, 1:round(len(weight_counter)/sum(weight_counter))}


  # Set optimizer ---------------------------------------------------------------------
  epochs = 10
  lr=0.0001
  lr_drop_factor = 0.75
  rmsprop = optimizers.RMSprop(lr)
  model.compile(optimizer=rmsprop,
                loss='binary_crossentropy',
                metrics=['accuracy'])


  # Train model on dataset ---------------------------------------------------------------
  history = {}
  loss_hist = {}
  save_filepath = ''
  for epoch in range(1,epochs+1):
    print('Epoch:{}/{}. Learning rate: {}'.format(epoch,epochs,lr))

    if epoch!=1:
      model = load_model(save_filepath)
      rmsprop = optimizers.RMSprop(lr)
      model.compile(optimizer=rmsprop,
                loss='binary_crossentropy',
                metrics=['accuracy']) 

    temp_hist = model.fit_generator(generator = training_generator,
                        epochs=1,
                        steps_per_epoch = len(partition['train'])//params['batch_size'],
                        class_weight=class_weight,
                        validation_data = validation_generator,
                        validation_steps = len(partition['validation'])//params['batch_size'],
                        callbacks = None,
                        max_queue_size=15,
                        workers=2)

    if epoch==1:
      save_filepath = 'fold-{}.epoch-{:02d}-loss-{:.3f}-acc-{:.3f}.plane-{}.hdf5'.format(test_fold,epoch,temp_hist.history['val_loss'][0],temp_hist.history['val_acc'][0],plane)
      model.save(save_filepath)
      history['{}'.format(epoch)] = temp_hist.history
      loss_hist['{}'.format(epoch)] = temp_hist.history['val_loss']
    elif temp_hist.history['val_loss'] < history['{}'.format(min(loss_hist, key=loss_hist.get))]['val_loss']:
      save_filepath = 'fold-{}.epoch-{:02d}-loss-{:.3f}-acc-{:.3f}.plane-{}.hdf5'.format(test_fold,epoch,temp_hist.history['val_loss'][0],temp_hist.history['val_acc'][0],plane)
      model.save(save_filepath)
      history['{}'.format(epoch)] = temp_hist.history
      loss_hist['{}'.format(epoch)] = temp_hist.history['val_loss']
    else:
      lr = lr*lr_drop_factor


  # # Save training history ----------------------------------------------------------------
  with open(path_history, 'wb') as file_pi:
      pickle.dump(history, file_pi)

  K.clear_session()