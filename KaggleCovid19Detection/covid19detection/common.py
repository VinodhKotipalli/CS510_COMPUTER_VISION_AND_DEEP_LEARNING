import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys
import copy
import cv2

import types

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure


filedir = os.path.dirname(os.path.abspath(__file__))
filedir = os.path.dirname(filedir).replace('\\', '/').replace('C:', '')
sys.path.insert(0, filedir)

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inputdir = projdir.replace('\\', '/') + '/data/csv'
imagedir = projdir.replace('\\', '/') + '/data/images'

# Function to read Image and Study level CSV files, merge and preprocessing them in desired format
def read_dataset_csv(study_csv_path,image_csv_path,size_csv_path,study_columns=None,padding_value_xy=0.0,padding_value_wh=0.0,image_format='jpg'):
    df_image = pd.read_csv(image_csv_path)
    df_study = pd.read_csv(study_csv_path)
    df_size = pd.read_csv(size_csv_path)

    prefix = image_csv_path.split('/')[-1].split('_')[0]
    prefix_study = study_csv_path.split('/')[-1].split('_')[0]
    if prefix != prefix_study:
        print("Info: Image level prefix and Study Level prefix don't match. Going with Prefix(%s) from Path(%s) and ignoring the Path(%s)" %(prefix,image_csv_path,study_csv_path))

    
    
    if study_columns is not None:
        df_study.columns = study_columns
        
    df_study['StudyInstanceUID'] = df_study['id'].replace(to_replace='_study',value='',regex=True)
    
    df_merged = df_image.merge(df_study[df_study.columns.tolist()[1:]], on=['StudyInstanceUID'])
    
    df_merged['ImageInstanceUID'] = df_merged['id'].replace(to_replace='_image',value='',regex=True)
    df_merged['ImageBoxCount'] = df_merged['label'].str.split(' ').str.len().divide(6)
    df_merged.loc[df_merged['boxes'].isna(),'ImageBoxCount'] = 0.0
    
    df_merged['fname'] = df_merged['id'].str.replace('_image', '.' + image_format)
    df_size = df_size.rename(columns={"id":"fname"})

    df_merged = df_merged.merge(df_size[['fname','dim0','dim1']], on=['fname'])
    max_boxes = int(df_merged['ImageBoxCount'].max())
    
    for i in range(max_boxes):
        df_merged['X'+str(i+1)] = df_merged['label'].str.split(' ').str[(6*i)+2].fillna(0.0).astype(float)
        df_merged['Y'+str(i+1)] = df_merged['label'].str.split(' ').str[(6*i)+3].fillna(0.0).astype(float)
        df_merged['W'+str(i+1)] = df_merged['label'].str.split(' ').str[(6*i)+4].fillna(0.0).astype(float) - df_merged['X'+str(i+1)]
        df_merged['H'+str(i+1)] = df_merged['label'].str.split(' ').str[(6*i)+5].fillna(0.0).astype(float) - df_merged['Y'+str(i+1)]
        
        df_merged.loc[df_merged['ImageBoxCount'] < i+1, 'X'+str(i+1)] = padding_value_xy
        df_merged.loc[df_merged['ImageBoxCount'] < i+1, 'Y'+str(i+1)] = padding_value_xy
        df_merged.loc[df_merged['ImageBoxCount'] < i+1, 'W'+str(i+1)] = padding_value_wh
        df_merged.loc[df_merged['ImageBoxCount'] < i+1, 'H'+str(i+1)] = padding_value_wh

        df_merged['X'+str(i+1)] = df_merged['X'+str(i+1)] / df_merged['dim0'] 
        df_merged['Y'+str(i+1)] = df_merged['X'+str(i+1)] / df_merged['dim1']
        df_merged['W'+str(i+1)] = df_merged['X'+str(i+1)] / df_merged['dim0']
        df_merged['H'+str(i+1)] = df_merged['X'+str(i+1)] / df_merged['dim1']
        
    df_merged = df_merged.fillna(0.0)
    study_labels = ['Negative','Typical','Indeterminate','Atypical']
    df_merged['study_label'] = 'UnAssigned'
    i = 0
    for column in df_study.columns.tolist()[1:-1]:
        df_merged.loc[df_merged[column]==1, 'study_label'] = study_labels[i]
        i += 1
    

    df_merged = df_merged.rename(columns={"label":"image_label"})
    return df_merged


# Function to read image in dicom format and convert to a array (Code resuse from Sina)
def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

# Function for image pre-processing (Code resuse from Sina)
def preprocess_image(img):
    equ_img = exposure.equalize_hist(img)
    return equ_img


# Function for reading image from the directory
def get_image_generator(inputdf,dirpath,x_col,y_col,img_x,img_y,batch_size,validation_split,subset):

    if type(y_col) is not list or len(y_col) == 1:
        class_mode='categorical'
    else:
        class_mode = 'multi_output'

    if subset != 'validation':
        image_generator = ImageDataGenerator(
            validation_split=validation_split,
            #rotation_range=20,
            horizontal_flip = True,
            zoom_range = 0.1,
            #shear_range = 0.1,
            brightness_range = [0.8, 1.1],
            fill_mode='nearest',
            preprocessing_function=preprocess_image,
        )
    else:
        image_generator = ImageDataGenerator(
            validation_split=validation_split,
            preprocessing_function=preprocess_image,
        )

    dataset = image_generator.flow_from_dataframe(
        dataframe = inputdf,
        directory=dirpath,
        x_col = x_col,
        y_col =  y_col,  
        target_size=(img_x,img_y),
        batch_size=batch_size,
        subset= subset,
        class_mode=class_mode
    ) 
    
    return dataset



if __name__ == '__main__':
    study_csv_path_training = inputdir+'/train_study_level.csv'
    image_csv_path_training = inputdir+'/train_image_level.csv'
    size_csv_path_training = imagedir+'/size.csv'
    study_columns = ['id','Negative','Typical','Indeterminate','Atypical']
    padding_value_xy = 0.0
    padding_value_wh = 0.0
    validation_split = 0.2
    steps_per_execution = 1
    num_hidden_0 = 256
    num_hidden_1 = 64
    num_hidden_2 = 16    
    
    
    img_x = 224
    img_y = 224
    batch_size = 16
    image_format = 'jpg'
    
    df_train = read_dataset_csv(study_csv_path_training,image_csv_path_training,size_csv_path_training,study_columns,padding_value_xy,padding_value_wh,image_format)

    max_boxes = int(df_train['ImageBoxCount'].max())

    y_col = study_columns[1:]
    for i in range(max_boxes):
        y_col.append('X'+str(i+1))
        y_col.append('Y'+str(i+1))
        y_col.append('W'+str(i+1))
        y_col.append('H'+str(i+1))

    generator_train = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'training')
    generator_val = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'validation')

    actual = np.array(generator_val.labels)
    print(y_col)
    print(len(actual[:,1]))