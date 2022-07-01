import os
import pandas as pd
import shutil

basepath = 'alldata/combined_images' #Origin of the images
# # basepath = 'new_data/real_image/real_combined' #Origin of the images
# train_destination = "alldata/real_train_images" #Destination of the trainning images
# test_destination = "alldata/real_test_images" #Destination of the testing images

# train_annotations = pd.read_csv('alldata/train_metadata.csv') #training CSV list
# # train_annotations = pd.read_csv('new_data/real_meta/real_meta_train.csv') #training CSV list
# test_annotations = pd.read_csv('alldata/test_metadata.csv') #testing CSV list
# # test_annotations = pd.read_csv('new_data/real_meta/real_meta_test.csv') #testing CSV list

# train_pathlist = train_annotations.iloc[:,0] #copy the targets columns only
# test_pathlist =  test_annotations.iloc[:, 0] #Copy the target column oly


#Train data
pneu_train_annotations = pd.read_csv('alldata/pneu_train_metadata.csv') 
norm_train_annotations = pd.read_csv('alldata/norm_train_metadata.csv')
pneu_test_annotations = pd.read_csv('alldata/pneu_test_metadata.csv') #training CSV list
norm_test_annotations = pd.read_csv('alldata/norm_test_metadata.csv') #training CSV list

pneu_train_destination = "alldata/pneu_train_images" 
norm_train_destination = "alldata/norm_train_images" 
pneu_test_destination = "alldata/pneu_test_images" 
norm_test_destination = "alldata/norm_test_images" 

pneu_train_pathlist = pneu_train_annotations.iloc[:,0] #copy the targets columns only
norm_train_pathlist = norm_train_annotations.iloc[:,0] #copy the targets columns only
pneu_test_pathlist =  pneu_test_annotations.iloc[:, 0] #Copy the target column oly
norm_test_pathlist =  norm_test_annotations.iloc[:, 0] #Copy the target column oly

#Set the type of images required
train_annotations = False

#read images scan all images and save in names in list
with os.scandir(basepath) as images:
    list=[]
    for image in images:
        list.append(image.name)

#label index list
if train_annotations is True:
    """Copy all images to train destination for peumonia and normal"""
    os.makedirs(pneu_train_destination, exist_ok=True)
    for path in pneu_train_pathlist:
        if path in list:
            # shutil.copytree('images', 'sifted2')
            shutil.copy(f'{basepath}/{path}', f"{pneu_train_destination}/{path}")
        print(f"File copied pneumonia train image {path} successfully.")
    
    os.makedirs(norm_train_destination, exist_ok=True)
    for path in norm_train_pathlist:
        if path in list:
            # shutil.copytree('images', 'sifted2')
            shutil.copy(f'{basepath}/{path}', f"{norm_train_destination}/{path}")
        print(f"File copied normal train image {path} successfully.")
    print("Copying train images completed")
else:
    """Copy all images to test destination"""
    os.makedirs(pneu_test_destination, exist_ok=True)
    for path in pneu_test_pathlist:
        if path in list:
            # shutil.copytree('images', 'sifted2')
            shutil.copy(f'{basepath}/{path}', f"{pneu_test_destination}/{path}")
        print(f"File copied pneumonia test image {path}  successfully.")

    os.makedirs(norm_test_destination, exist_ok=True)
    for path in norm_test_pathlist:
        if path in list:
            # shutil.copytree('images', 'sifted2')
            shutil.copy(f'{basepath}/{path}', f"{norm_test_destination}/{path}")
        print(f"File copied normal test image {path}  successfully.")
    print("Copying test images completed")


