# CONSTANTS --------------------------------------------------------------------------------------
DNN_INPUT_SIZE = 40 # (40x40x40) volume
SAMPLE_DIM = 44 # dimensions of the extracted subvolume sample
SAMPLE_PADDING = SAMPLE_DIM-DNN_INPUT_SIZE # SAMPLE_PADDING/2 voxel padding in each direction of sample volume size

# # Extract the sample volumes ---------------------------------------------------------------------
# import tools
# import time

# start_time = time.time()
# sample_type = 'annotations'
# csv_dir = 'D:/annotations.csv'
# scan_dir = 'D:/raw_data'
# save_dir = 'D:/Lung_Data/extracted_volumes/positives_dilated'
# tools.extract_samples(sample_type, SAMPLE_DIM, csv_dir, scan_dir, save_dir)
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# sample_type = 'candidates'
# csv_dir = 'D:/candidates.csv'
# scan_dir = 'D:/raw_data'
# save_dir = 'D:/Lung_Data/extracted_volumes/negatives'
# tools.extract_samples(sample_type, DNN_INPUT_SIZE, csv_dir, scan_dir, save_dir)
# print("--- %s seconds ---" % (time.time() - start_time))


# # Augment the samples for training ---------------------------------------------------------------
# import tools
# import time

# sample_dir = 'D:/Lung_Data/extracted_volumes/positives_dilated'
# save_dir = 'D:/Lung_Data/extracted_volumes/positives_base'
# start_time = time.time()
# tools.extract_testing_samples(sample_dir, save_dir, SAMPLE_PADDING)
# print("--- %s seconds ---" % (time.time() - start_time))

# sample_dir = 'D:/Lung_Data/extracted_volumes/positives_dilated'
# save_dir = 'D:/Lung_Data/extracted_volumes/positives_augmented'
# start_time = time.time()
# tools.augment_samples(sample_dir, save_dir, DNN_INPUT_SIZE)
# print("--- %s seconds ---" % (time.time() - start_time))


# # Viewing augmented samples ---------------------------------------------------------------------

import tools
filepath = 'D:\\Lung_Data\\extracted_volumes\\negatives\\subset9\\candidate_csv_index_125160.npy'
tools.sample_viewer(filepath,rotate=False)