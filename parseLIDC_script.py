# Examples of annotation features and their meanings:
# 
# ann.print_formatted_feature_table()
# => Feature              Meaning                    # 
# => -                    -                          - 
# => Subtlety           | Obvious                  | 5 
# => Internalstructure  | Soft Tissue              | 1 
# => Calcification      | Absent                   | 6 
# => Sphericity         | Ovoid                    | 3 
# => Margin             | Poorly Defined           | 1 
# => Lobulation         | Near Marked Lobulation   | 4 
# => Spiculation        | Medium Spiculation       | 3 
# => Texture            | Solid                    | 5 
# => Malignancy         | Moderately Suspicious    | 4 -------> Malignancy is rated on a 5-point scale, 5 being most suspicious
# -----------------------------------------------------------------------------------------------------------------------------


## Store the origin and spacing of each scan. This is a temporary fix for the coordinate issue in the pyLIDC library
## The pyLIDC library returns (x,y,z) coordinates where (x,y) are voxel coordinates and z is a world coordinate.
# import SimpleITK as sitk
# import glob
# import os.path
# import pickle
# import csv

# print('Extracting Scan Image Info...')
# imageInfo = list()
# scan_num = 1
# for ii in range(0,10): # store filenames in each subset
#     filepath = 'E:/Luna/raw_data/subset{}/*.mhd'.format(ii)
#     mhd_files = glob.glob(filepath)
#     for row in mhd_files:
#         itkimage = sitk.ReadImage(row)
#         origin = itkimage.GetOrigin()
#         spacing = itkimage.GetSpacing()
#         imageInfo.append((os.path.basename(row),origin,spacing))
#         print(scan_num)
#         scan_num += 1
# with open('E:\\LIDC\\scan_origin_info.txt', 'wb') as fp:   #Pickling
#     pickle.dump(imageInfo, fp)



# csv_existing_file = 'E:\\LIDC\\annotations.csv'
# csv_save_file = 'E:\\LIDC\\annotations_voxel_coords.csv'

# # Open saved file that contains the scan image info (seriesUID, origin, spacing)
# with open('E:\\LIDC\\scan_origin_info.txt', 'rb') as fp:   # Unpickling
#     imageInfo = pickle.load(fp)

# # Read existing annotations csv file and store info as list
# with open(csv_existing_file, newline='') as csvfile:
#     samples = csv.reader(csvfile, delimiter=',', quotechar='|')
#     samples_list = list(samples)
#     samples_list.pop(0)

# with open(csv_save_file, 'w', newline='') as myfile:
#     wr = csv.writer(myfile, delimiter=',', lineterminator='\n')
#     wr.writerow(['seriesUID', 'worldCoordX', 'worldCoordY', 'worldCoordZ', 'voxCoordX', 'voxCoordY', 'voxCoordZ', 'avg_diameter'])

#     for row in samples_list:
#         worldCoordX = float(row[1])
#         worldCoordY = float(row[2])
#         worldCoordZ = float(row[3])

#         matches = [y for y, x in enumerate(imageInfo) if (x[0][:-4]==row[0])]
#         assert len(matches)==1,'Too many matches.'
#         origin = imageInfo[matches[0]][1]
#         spacing = imageInfo[matches[0]][2]

#         voxCoordX = abs(worldCoordX-float(origin[0]))/float(spacing[0])
#         voxCoordY = abs(worldCoordY-float(origin[1]))/float(spacing[1])
#         voxCoordZ = abs(worldCoordY-float(origin[2]))/float(spacing[2])

#         row_list = [row[0], worldCoordX, worldCoordY, worldCoordZ, voxCoordX, voxCoordY, voxCoordZ, row[4]]
#         print(row_list)
#         wr.writerow(row_list)
        # myfile.flush()

# -----------------------------------------------------------------------------------------------------------------------------

# Define csv file names (existing and save). Open the existing annotation csv file.
import pylidc as pl
import csv


csv_existing_file = 'E:\\LIDC\\annotations_voxel_coords.csv'
csv_save_file = 'E:\\LIDC\\annotations_enhanced.csv'

# Read existing annotations csv file and store info as list
with open(csv_existing_file, newline='') as csvfile:
    samples = csv.reader(csvfile, delimiter=',', quotechar='|')
    samples_list = list(samples)
    samples_list.pop(0)

# Filter the scans, collect only those with spacing <= 2.5
filtered_scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 2.5)


# Create new annotations csv file with esixting data + malignancy data
with open(csv_save_file, 'w', newline='') as myfile:
    wr = csv.writer(myfile, delimiter=',', quotechar='|')
    wr.writerow(['seriesUID', 'worldCoordX', 'worldCoordY', 'worldCoordZ', 'voxCoordX', 'voxCoordY', 'voxCoordZ', 'avg_diameter', 'malignancy_1', 'malignancy_2', 'malignancy_3', 'malignancy_4', 'avg_malignancy'])
    
    temp_list =[]
    for counter in range(0,1):
        if counter==0:
            metric='jaccard'
        elif counter==1:
            metric='centroid_xyz'
        elif counter==2:
            metric='hausdorff'
        elif counter==3:
            metric='min'
        for scan in filtered_scans:
            nods = scan.cluster_annotations(metric, tol=None, factor=0.9)
            print("Scan ",scan.series_instance_uid, " is estimated to have", len(nods), "nodules.")
            for ii,nod in enumerate(nods):
                print("Nodule", ii+1, "has", len(nod), "annotations.")
                if (len(nod) >= 3):
                    avg_diameter = []
                    avg_coordX = []
                    avg_coordY = []
                    avg_coordZ = []
                    malignancy = [None, None, None, None]

                    for jj,ann in enumerate(nod):
                        avg_diameter.append(ann.estimate_diameter())
                        coordinates = ann.centroid(image_coords=True)
                        avg_coordX.append(coordinates[0])
                        avg_coordY.append(coordinates[1])
                        avg_coordZ.append(coordinates[2])
                        malignancy[jj] = ann.malignancy
                    avg_coordX = sum(avg_coordX)/float(len(nod))
                    avg_coordY = sum(avg_coordY)/float(len(nod))
                    avg_coordZ = sum(avg_coordZ)/float(len(nod))
                    avg_diameter = sum(avg_diameter)/float(len(nod))

                    uid_matches = [y for y, x in enumerate(samples_list) if (x[0]==scan.series_instance_uid)]
                    data_matches = [y for y, x in enumerate(samples_list) if ( abs(float(x[4])-avg_coordX)  <= 10 ) and ( abs(float(x[5])-avg_coordY)  <= 10 )  and ( abs(float(x[3])-avg_coordZ)  <= 10 )]
                    matches = set(uid_matches) & set(data_matches)

                    if len(matches)==1:
                        match = list(matches)[0]
                        temp_list.append([scan.series_instance_uid, samples_list[match][1], samples_list[match][2], samples_list[match][3], samples_list[match][4], samples_list[match][5], samples_list[match][6], samples_list[match][7], malignancy[0], malignancy[1], malignancy[2], malignancy[3]])
                    else:
                        assert 'Too many matches.'
    print(len(temp_list))
    combined_list = [list(item) for item in set(tuple(row) for row in temp_list)]
    print(len(combined_list))

    wr.writerows(combined_list)
    myfile.flush()