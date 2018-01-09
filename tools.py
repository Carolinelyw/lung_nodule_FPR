# SAMPLE VIEWER TOOL --------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def sample_viewer(func_input,rotate=False):
    if isinstance(func_input, str):
        sample = np.load(func_input)
    else:
        sample = func_input
    if rotate==True:
        sample = np.rot90(sample,1,(2,0))
    print(sample.shape)
    for disp_ctr in range(-int(sample.shape[0]/2-1),int(sample.shape[0]/2-1)):
        plt.imshow(sample[int(sample.shape[0]/2+disp_ctr), :, :],cmap='gray',interpolation='nearest')
        plt.show()


# DATAPREP TOOLS -----------------------------------------------------------------------------------------------------------------------
import numpy as np
import csv
import glob
import SimpleITK as sitk
import scipy.ndimage.interpolation as interp

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord.astype(int)

def normalizeIntensity(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def extract_samples(sample_type, sample_dim, csv_dir, scan_dir, save_dir):
    print('Extracting sample volumes...')

    with open(csv_dir, newline='') as csvfile:
        samples = csv.reader(csvfile, delimiter=',', quotechar='|')
        samples_list = list(samples)
    seriesuidS = [ii[0] for ii in samples_list]
    lesion_type = [i[4] for i in samples_list]

    save_num = 1
    for subset_ctr in range(0,10):
        fold = '{}/subset{}'.format(scan_dir, str(subset_ctr))
        for filename in glob.glob(fold+'/*.mhd'):
            seriesuid = filename[-68:-4]
            ct, origin, spacings = load_itk_image(filename)

            matches = [y for y, x in enumerate(seriesuidS) if x == seriesuid]
            for match in matches:
                if (sample_type=='candidates' and lesion_type[match]=='0') or (sample_type=='annotations'):
                    ct2 = ct
                    worldCoord = float(samples_list[match][3]), float(samples_list[match][2]), float(samples_list[match][1]) # (z,y,x) coordinates
                    voxelCoord = worldToVoxelCoord(worldCoord,origin,spacings)
                    
                    if sample_type == 'candidates':
                        save_name = 'candidate_csv_index_{}'.format(match)
                    else:
                        save_name = 'annotation_csv_index_{}'.format(match)

                    dim_x = int(round(22/spacings[2]))
                    dim_y = int(round(22/spacings[1]))
                    dim_z = int(round(22/spacings[0]))

                    if (voxelCoord[0] < dim_z) or (voxelCoord[0]+dim_z > ct.shape[0]) or (voxelCoord[1] < dim_y) or (voxelCoord[1]+dim_y > ct.shape[1]) or (voxelCoord[2] < dim_x) or (voxelCoord[2]+dim_x > ct.shape[2]):
                        extendby = max(dim_x,dim_y,dim_z)
                        voxelCoord = voxelCoord + extendby
                        padded = np.zeros((ct.shape[0]+extendby*2,ct.shape[1]+extendby*2,ct.shape[2]+extendby*2))
                        padded[extendby:ct.shape[0]+extendby, extendby:ct.shape[1]+extendby, extendby:ct.shape[2]+extendby] = ct
                        ct2 = padded
                        print('padded...')

                    cuboid = ct2[voxelCoord[0]-dim_z:voxelCoord[0]+dim_z,voxelCoord[1]-dim_y:voxelCoord[1]+dim_y,voxelCoord[2]-dim_x:voxelCoord[2]+dim_x]
                    # cuboid = normalizeIntensity(cuboid)
                    cuboid = interp.zoom( cuboid,(sample_dim/cuboid.shape[0],sample_dim/cuboid.shape[1],sample_dim/cuboid.shape[2]) )

                    # sample_viewer(cuboid)

                    np.save('{}/subset{}/{}'.format(save_dir,subset_ctr,save_name), cuboid)
                    print('Save #{}'.format(save_num))
                    save_num += 1

    print('Sample volumes have been saved.')


# DATA AUGMENTATION TOOLS --------------------------------------------------------------------------------------------------------------
import numpy as np
import glob
import os.path

def  rotateVolume(arrBase,degrees):
    assert (degrees==90 or degrees==180 or degrees==270),'Not a valid degree of rotation. Choose 90, 180, or 270.'

    if degrees == 90:
        output = np.rot90(arrBase,1,(1,2))
    elif degrees == 180:
        output = np.rot90(arrBase,2,(1,2))
    elif degrees == 270:
        output = np.rot90(arrBase,3,(1,2))
    return  output

def flipVolume(arrBase,axis):
    assert (axis=='y' or axis=='x'),'Not a valid axis of rotation. Choose ''x'' or ''y''.'

    if axis == 'y':
        output = np.rot90(arrBase,2,(1,0))
    elif axis == 'x':
        output = np.rot90(arrBase,2,(2,0))   
    return output

def cropBaseVolume(arrPadded,padding):
    '''Center crops a (Z+padding,Y+padding,X+padding) numpy array to (Z,Y,X). Padding should be an even integer.'''
    padding = int(padding/2)
    arrBase = arrPadded[padding:-padding,padding:-padding,padding:-padding]
    return arrBase

def translateXYZ(arrPadded,direction,input_size,translate_by):
    assert (translate_by*2 <= (arrPadded.shape[0]-input_size)),'Translation out of bounds! Try translating by a smaller value.'

    margin = int((arrPadded.shape[0]-input_size)/2)
    if (-margin+translate_by) == 0:
        margin_w_expection = arrPadded.shape[0]
    else:
        margin_w_expection = -margin+translate_by

    translatedArr = np.zeros((input_size,input_size,input_size))
    if direction == 'down':
        translatedArr = arrPadded[margin:-margin, margin-translate_by:-margin-translate_by, margin:-margin]
    elif direction == 'up':
        translatedArr = arrPadded[margin:-margin, margin+translate_by:margin_w_expection, margin:-margin]
    elif direction == 'right':
        translatedArr = arrPadded[margin:-margin, margin:-margin, margin-translate_by:-margin-translate_by]
    elif direction == 'left':
        translatedArr = arrPadded[margin:-margin, margin:-margin, margin+translate_by:margin_w_expection]
    elif direction == 'forward':
        translatedArr = arrPadded[margin+translate_by:margin_w_expection, margin:-margin, margin:-margin]
    elif direction == 'backward':
        translatedArr = arrPadded[margin-translate_by:-margin-translate_by, margin:-margin, margin:-margin]
    else:
        print('Not a valid direction.')
    return translatedArr

def augmentation_helper(augmentation_num,input_size,arrPadded,arrBase):
    # All possible augmentations:

    if augmentation_num == 1:
        translate_by = 2
        output = translateXYZ(arrPadded,'left',input_size,translate_by)
    elif augmentation_num == 2:
        translate_by = 2
        output = translateXYZ(arrPadded,'right',input_size,translate_by)
    elif augmentation_num == 3:
        translate_by = 2
        output = translateXYZ(arrPadded,'up',input_size,translate_by)
    elif augmentation_num == 4:
        translate_by = 2
        output = translateXYZ(arrPadded,'down',input_size,translate_by)
    elif augmentation_num == 5:
        translate_by = 2
        output = translateXYZ(arrPadded,'forward',input_size,translate_by)
    elif augmentation_num == 6:
        translate_by = 2
        output = translateXYZ(arrPadded,'backward',input_size,translate_by)

    elif augmentation_num == 7:
        output = flipVolume(arrBase,'x')
    elif augmentation_num == 8:
        output = flipVolume(arrBase,'y')

    elif augmentation_num == 9:
        output = rotateVolume(arrBase,90)
    elif augmentation_num == 10:
        output = rotateVolume(arrBase,180)
    elif augmentation_num == 11:
        output = rotateVolume(arrBase,270)

    elif augmentation_num == 12:
        translate_by = 2
        output = translateXYZ(arrPadded,'left',input_size,translate_by)
        output = rotateVolume(output,90)
    elif augmentation_num == 13:
        translate_by = 2
        output = translateXYZ(arrPadded,'right',input_size,translate_by)
        output = rotateVolume(output,90)
    elif augmentation_num == 14:
        translate_by = 2
        output = translateXYZ(arrPadded,'up',input_size,translate_by)
        output = rotateVolume(output,90)
    elif augmentation_num == 15:
        translate_by = 2
        output = translateXYZ(arrPadded,'down',input_size,translate_by)
        output = rotateVolume(output,90)
    elif augmentation_num == 16:
        translate_by = 2
        output = translateXYZ(arrPadded,'forward',input_size,translate_by)
        output = rotateVolume(output,90)
    elif augmentation_num == 17:
        translate_by = 2
        output = translateXYZ(arrPadded,'backward',input_size,translate_by)
        output = rotateVolume(output,90)

    elif augmentation_num == 18:
        translate_by = 2
        output = translateXYZ(arrPadded,'left',input_size,translate_by)
        output = rotateVolume(output,180)
    elif augmentation_num == 19:
        translate_by = 2
        output = translateXYZ(arrPadded,'right',input_size,translate_by)
        output = rotateVolume(output,180)
    elif augmentation_num == 20:
        translate_by = 2
        output = translateXYZ(arrPadded,'up',input_size,translate_by)
        output = rotateVolume(output,180)
    elif augmentation_num == 21:
        translate_by = 2
        output = translateXYZ(arrPadded,'down',input_size,translate_by)
        output = rotateVolume(output,180)
    elif augmentation_num == 22:
        translate_by = 2
        output = translateXYZ(arrPadded,'forward',input_size,translate_by)
        output = rotateVolume(output,180)
    elif augmentation_num == 23:
        translate_by = 2
        output = translateXYZ(arrPadded,'backward',input_size,translate_by)
        output = rotateVolume(output,180)

    elif augmentation_num == 24:
        translate_by = 2
        output = translateXYZ(arrPadded,'left',input_size,translate_by)
        output = rotateVolume(output,270)
    elif augmentation_num == 25:
        translate_by = 2
        output = translateXYZ(arrPadded,'right',input_size,translate_by)
        output = rotateVolume(output,270)
    elif augmentation_num == 26:
        translate_by = 2
        output = translateXYZ(arrPadded,'up',input_size,translate_by)
        output = rotateVolume(output,270)
    elif augmentation_num == 27:
        translate_by = 2
        output = translateXYZ(arrPadded,'down',input_size,translate_by)
        output = rotateVolume(output,270)
    elif augmentation_num == 28:
        translate_by = 2
        output = translateXYZ(arrPadded,'forward',input_size,translate_by)
        output = rotateVolume(output,270)
    elif augmentation_num == 29:
        translate_by = 2
        output = translateXYZ(arrPadded,'backward',input_size,translate_by)
        output = rotateVolume(output,270)

    elif augmentation_num == 30:
        translate_by = 2
        output = translateXYZ(arrPadded,'left',input_size,translate_by)
        output = flipVolume(output,'x')
    elif augmentation_num == 31:
        translate_by = 2
        output = translateXYZ(arrPadded,'right',input_size,translate_by)
        output = flipVolume(output,'x')
    elif augmentation_num == 32:
        translate_by = 2
        output = translateXYZ(arrPadded,'up',input_size,translate_by)
        output = flipVolume(output,'x')
    elif augmentation_num == 33:
        translate_by = 2
        output = translateXYZ(arrPadded,'down',input_size,translate_by)
        output = flipVolume(output,'x')
    elif augmentation_num == 34:
        translate_by = 2
        output = translateXYZ(arrPadded,'forward',input_size,translate_by)
        output = flipVolume(output,'x')
    elif augmentation_num == 35:
        translate_by = 2
        output = translateXYZ(arrPadded,'backward',input_size,translate_by)
        output = flipVolume(output,'x')

    elif augmentation_num == 36:
        translate_by = 2
        output = translateXYZ(arrPadded,'left',input_size,translate_by)
        output = flipVolume(output,'y')
    elif augmentation_num == 37:
        translate_by = 2
        output = translateXYZ(arrPadded,'right',input_size,translate_by)
        output = flipVolume(output,'y')
    elif augmentation_num == 38:
        translate_by = 2
        output = translateXYZ(arrPadded,'up',input_size,translate_by)
        output = flipVolume(output,'y')
    elif augmentation_num == 39:
        translate_by = 2
        output = translateXYZ(arrPadded,'down',input_size,translate_by)
        output = flipVolume(output,'y')
    elif augmentation_num == 40:
        translate_by = 2
        output = translateXYZ(arrPadded,'forward',input_size,translate_by)
        output = flipVolume(output,'y')
    elif augmentation_num == 41:
        translate_by = 2
        output = translateXYZ(arrPadded,'backward',input_size,translate_by)
        output = flipVolume(output,'y')

    elif augmentation_num == 42:
        output = rotateVolume(arrBase,90)
        output = flipVolume(output,'x')
    elif augmentation_num == 43:
        output = rotateVolume(arrBase,90)
        output = flipVolume(output,'x')
    elif augmentation_num == 44:
        output = rotateVolume(arrBase,90)
        output = flipVolume(output,'x')
    elif augmentation_num == 45:
        output = rotateVolume(arrBase,90)
        output = flipVolume(output,'y')
    elif augmentation_num == 46:
        output = rotateVolume(arrBase,90)
        output = flipVolume(output,'y')
    elif augmentation_num == 47:
        output = rotateVolume(arrBase,90)
        output = flipVolume(output,'y')

    elif augmentation_num == 48:
        output = rotateVolume(arrBase,180)
        output = flipVolume(output,'x')
    elif augmentation_num == 49:
        output = rotateVolume(arrBase,180)
        output = flipVolume(output,'x')
    elif augmentation_num == 50:
        output = rotateVolume(arrBase,180)
        output = flipVolume(output,'x')
    elif augmentation_num == 51:
        output = rotateVolume(arrBase,180)
        output = flipVolume(output,'y')
    elif augmentation_num == 52:
        output = rotateVolume(arrBase,180)
        output = flipVolume(output,'y')
    elif augmentation_num == 53:
        output = rotateVolume(arrBase,180)
        output = flipVolume(output,'y')

    elif augmentation_num == 54:
        output = rotateVolume(arrBase,270)
        output = flipVolume(output,'x')
    elif augmentation_num == 55:
        output = rotateVolume(arrBase,270)
        output = flipVolume(output,'x')
    elif augmentation_num == 56:
        output = rotateVolume(arrBase,270)
        output = flipVolume(output,'x')
    elif augmentation_num == 57:
        output = rotateVolume(arrBase,270)
        output = flipVolume(output,'y')
    elif augmentation_num == 58:
        output = rotateVolume(arrBase,270)
        output = flipVolume(output,'y')
    elif augmentation_num == 59:
        output = rotateVolume(arrBase,270)
        output = flipVolume(output,'y')

    return output


def extract_testing_samples(sample_dir,save_dir,padding):
    '''Used for extracting a dataset of un-augmented positive samples with the expected dimensions of the DNN as an input.
        Dataset will then be used for testing purposes.'''
    save_num = 1
    for subset_ctr in range(0,10):
        fold = '{}/subset{}/'.format(sample_dir, str(subset_ctr))
        for filename in glob.glob(fold+'/*.npy'):
            arrPadded = np.load(filename)
            filename = os.path.basename(filename)
            filename = filename[:-4]            

            # extract the base volume
            count = '0' # for appending to end of image save name
            arrBase = cropBaseVolume(arrPadded,padding)
            np.save('{}/subset{}/{}_{}'.format( save_dir,subset_ctr,filename,count.zfill(2) ), arrBase)
            count = str(int(count)+1)

            print('Save #{}'.format(save_num))
            save_num += 1
    print('Base sample volumes have been created and saved.')

def augment_samples(sample_dir,save_dir,input_size):
    save_num = 1
    for subset_ctr in range(0,10):
        fold = '{}/subset{}'.format(sample_dir, str(subset_ctr))
        for filename in glob.glob(fold+'/*.npy'):
            arrPadded = np.load(filename)
            filename = os.path.basename(filename)
            filename = filename[:-4]

            padding = arrPadded.shape[0]-input_size

            count = '0' # for appending to end of image save name

            # extract the base volume, not augmented
            arrBase = cropBaseVolume(arrPadded,padding)
            np.save('{}/subset{}/{}_{}'.format( save_dir,subset_ctr,filename,count.zfill(2) ), arrBase)
            count = str(int(count)+1)

            # perform all other desired augmentations. can randomize the values in augmentations_list to do random augmentations
            augmentation_list = range(1,60)
            for augmentation_num in augmentation_list:
                arrAugmented = augmentation_helper(augmentation_num,input_size,arrPadded,arrBase)              
                np.save('{}/subset{}/{}_{}'.format( save_dir,subset_ctr,filename,count.zfill(2) ), arrAugmented)
                count = str(int(count)+1)

            print('Sample save #{}'.format(save_num))
            save_num += 1
    print('Data augmentation complete.')



# THREADING TOOLS -----------------------------------------------------------------------------------------------------------------------
''' Source: Shervine Amidi @ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html '''
import threading

class threadsafe_iter(object):
  """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()
  def __iter__(self):
      return self
  def __next__(self):
      with self.lock:
          return self.it.__next__()
def threadsafe_generator(f):
  """
    A decorator that takes a generator function and makes it thread-safe.
    """
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g


# PARTITIONING DATA TOOLS -----------------------------------------------------------------------------------------------------------------------
import glob
from random import shuffle

def remove_percentage(list_a, percentage): # percentage is a float value between 0.0 and 1.0
    shuffle(list_a)
    count = int(len(list_a) * percentage)
    if not count: return []  # edge case, no elements removed
    list_a[-count:], list_b = [], list_a[-count:]
    return list_b

def partition_data(subset_range, test_fold, valid_fold, neg_fold_path, pos_fold_path_train, pos_fold_path_test):
    labels = {} # dict of labels for each filepath
    partition = {} # dict of 'train', 'test', and 'validation'
    trainList = list()
    testList = list()
    validList = list()
    
    # loop thru each negative folder subset
    for jj in subset_range:
        if jj == test_fold: # test data
            fold = '{}/subset{}'.format(neg_fold_path, str(jj))
            for filepath in glob.glob(fold+'/*.npy'):
                testList.append(filepath)
                labels['{}'.format(filepath)] = int(0) # class = non-nodule
        elif jj == valid_fold: # validation data
            fold = '{}/subset{}'.format(neg_fold_path, str(jj))
            for filepath in glob.glob(fold+'/*.npy'):
                validList.append(filepath)
                labels['{}'.format(filepath)] = int(0)
        else: # train data
            fold = '{}/subset{}'.format(neg_fold_path, str(jj))
            for filepath in glob.glob(fold+'/*.npy'):
                trainList.append(filepath)
                labels['{}'.format(filepath)] = int(0)

    # loop thru each positive folder subset
    for jj in subset_range:
        if jj == test_fold:
            fold = '{}/subset{}'.format(pos_fold_path_test, str(jj))
            for filepath in glob.glob(fold+'/*.npy'):
                testList.append(filepath)
                labels['{}'.format(filepath)] = int(1) # class = nodule
        elif jj == valid_fold:
            fold = '{}/subset{}'.format(pos_fold_path_test, str(jj))
            for filepath in glob.glob(fold+'/*.npy'):
                validList.append(filepath)
                labels['{}'.format(filepath)] = int(0)
        else:
            fold = '{}/subset{}'.format(pos_fold_path_train, str(jj))
            for filepath in glob.glob(fold+'/*.npy'):
                trainList.append(filepath)
                labels['{}'.format(filepath)] = int(1)
        
        # store partitions into dictionary
        partition['test'] = testList
        partition['train'] = trainList
        partition['validation'] = validList

    print('Dataset partitioned.')
    return partition, labels


# CONFUSION MATRIX TOOLS ------------------------------------------------------------------------------------------------------------
'''Source:  scikit-learn.org '''
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def create_confusion_matrix(test_labels,class_predictions):
    cm = confusion_matrix(test_labels, class_predictions)
    cm_plot_labels = ['non-nodule','nodule']
    plt.figure()
    plot_confusion_matrix(cm,cm_plot_labels,title='confusion_matrix')
    plt.show()
