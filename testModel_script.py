from DataGenerator import DataGenerator
from keras.models import load_model
import numpy as np
import tools
import os.path
import csv

from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)


# # Opening history file:
# import pickle
# with open('D:\\lung_test_results\\metrics_temporary', 'rb') as f:
#     data = pickle.load(f)
# print(data)


# Filepaths --------------------------------------------------------------------------------------------
path_neg_fold = 'D:\\Luna\\extracted_volumes\\negatives'
path_pos_fold_train = 'D:\\Luna\\extracted_volumes\\positives_augmented'
path_pos_fold_test = 'D:\\Luna\\extracted_volumes\\positives_base'
path_model = 'C:\\Users\\ben43\\Dropbox\\Ben Folder\\ECE 693 - Deep Learning in Medical Imaging\\Lung_Project\\fold-0.epoch-02-loss-0.069-acc-0.985.plane-xy.hdf5'

buildCSV = True
ann_csv_dir = 'D:\\Luna\\annotations.csv'
cand_csv_dir = 'D:\\Luna\\candidates.csv'
predictions_save_dir = 'D:\\predictions'

# Parameters --------------------------------------------------------------------------------------------
params = {'dim_x': 40,
          'dim_y': 40,
          'dim_z': 40,
          'batch_size': 40, # <<-----###
          'shuffle': False}
plane = 'xy'

# Datasets ----------------------------------------------------------------------------------------------
# [(9,8,40),(8,7,149),(7,6,55),(6,5,131),(5,4,1),(4,3,10),(3,2,79),(2,1,30),(1,0,6),(0,9,48)]: # for (test_fold,train_fold,batch_size)
subset_range = range(0,10)
test_fold = 0
valid_fold = 9
partition,labels = tools.partition_data(subset_range, test_fold, valid_fold, path_neg_fold, path_pos_fold_train, path_pos_fold_test)
print( 'Length of test set: {}'.format(len(partition['test'])) )


# Get list of test labels --------------------------------------------------------------------------------
test_labels = []
test_keys = []
for ii in range(0,len(partition['test'])):
    test_labels.append(labels[partition['test'][ii]])
    test_keys.append(os.path.basename(partition['test'][ii][:-4]))
test_labels = np.array(test_labels)

# Create data generator and load model ---------------------------------------------------------------------
testing_generator = DataGenerator(**params).generate(labels, partition['test'], plane)
model = load_model(path_model)

# Test model on dataset ------------------------------------------------------------------------------------
predictions = model.predict_generator(generator = testing_generator,
                    steps = len(partition['test'])//params['batch_size'],
                    verbose = 1,
                    max_queue_size=10,
                    workers=2)

# Prepare results for analysis -------------------------------------------------------------------------------
class_predictions = predictions.argmax(1)
test_labels = test_labels[:len(class_predictions)]
tools.create_confusion_matrix(test_labels,class_predictions)

# Create csv file with format: seriesUID, coordX, coordY, coordZ, probability ---------------------------------
if buildCSV == True:
    with open(cand_csv_dir, newline='') as csvfile:
        candidates = csv.reader(csvfile, delimiter=',', quotechar='|')
        cand_list = list(candidates)
    with open(ann_csv_dir, newline='') as csvfile:
        annotations = csv.reader(csvfile, delimiter=',', quotechar='|')
        ann_list = list(annotations)

    predictions_save_dir = '{}-{}.csv'.format(predictions_save_dir,test_fold) # rename save dir for predictions
    with open(predictions_save_dir, 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',', lineterminator='\n')
        wr.writerow(['extracted_vol_ID', 'seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])

        for ii in range(0,len(test_keys)):
            test_key_info = test_keys[ii].split('_')
            index = int(test_key_info[3])
            if test_key_info[0]=='candidate':
                predictions_row = [ test_keys[ii],cand_list[index][0],cand_list[index][1],cand_list[index][2],cand_list[index][3],predictions[ii][1] ]
            elif test_key_info[0]=='annotation':
                predictions_row = [ test_keys[ii],ann_list[index][0],ann_list[index][1],ann_list[index][2],ann_list[index][3],predictions[ii][1] ]
            wr.writerow(predictions_row)