import os
import pandas as pd
import shutil

basepath = 'new_data/real_image/real_combined' #Origin of the images
train_destination = "real_train_images" #Destination of the trainning images
test_destination = "real_test_images" #Destination of the testing images
train_annotations = pd.read_csv('new_data/real_meta/real_meta_train.csv') #training CSV list
test_annotations = pd.read_csv('new_data/real_meta/real_meta_test.csv') #testing CSV list
train_pathlist = train_annotations.iloc[:,0] #copy the targets columns only
test_pathlist =  test_annotations.iloc[:, 0] #Copy the target column oly

#Set the type of images required
train_annotations = False

#read images scan all images and save in names in list
with os.scandir(basepath) as images:
    list=[]
    for image in images:
        list.append(image.name)

#label index list
if train_annotations is True:
    """Copy all images to test destination"""
    os.makedirs(train_destination, exist_ok=True)
    for path in train_pathlist:
        if path in list:
            # shutil.copytree('images', 'sifted2')
            shutil.copy(f'{basepath}/{path}', f"{train_destination}/{path}")
        print("File copied train images successfully.")
else:
    """Copy all images to test destination"""
    os.makedirs(test_destination, exist_ok=True)
    for path in test_pathlist:
        if path in list:
            # shutil.copytree('images', 'sifted2')
            shutil.copy(f'{basepath}/{path}', f"{test_destination}/{path}")
        print("File copied test images successfully.")
