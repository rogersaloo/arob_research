##TRAIN KERAS DATASET
# example of calculating the frechet inception distance in Keras for cifar10
import matplotlib.pyplot as plt
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
from tensorflow import keras
from numpy.random import randint
from keras.preprocessing.image import ImageDataGenerator
import sys
import fid_config

#Load synthetic and real images
image_size=(256,256)
batch_size=10
dataset_path_normal_real = fid_config.NORMAL_PATH
dataset_path_normal_synth = fid_config.SYNTHESIS_PATH

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

real_batches = train_datagen.flow_from_directory(
	dataset_path_normal_real,
	target_size=image_size,
	batch_size=batch_size,
	class_mode='binary',
	subset='training'
)

synth_batches = train_datagen.flow_from_directory(
	dataset_path_normal_synth,
	target_size=image_size,
	batch_size=batch_size,
	class_mode='binary',
	subset='validation'
)

images1=real_batches
images2=synth_batches

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)



# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)


#Load the training data
# image_size=(256,256)
# batch_size=10
# dataset_path = "data/train/"
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
#
# train_batches = train_datagen.flow_from_directory(
# 	dataset_path,
# 	target_size=image_size,
# 	batch_size=batch_size,
# 	class_mode='binary',
# 	subset='training'
# )
#
# validation_batches = train_datagen.flow_from_directory(
# 	dataset_path,
# 	target_size=image_size,
# 	batch_size=batch_size,
# 	class_mode='binary',
# 	subset='validation'
# )
#
# test_batches = train_datagen.flow_from_directory(
# 	dataset_path,
# 	target_size=image_size,
# 	batch_size=batch_size,
# 	class_mode='binary',
# 	shuffle=False,
# 	subset='validation'
# )

#Plot the images
# def plotImages(images_arr):
# 	fig, axes = plt.subplots(1,5)
# 	axes = axes.flatten()
# 	for img, ax in zip(images_arr, axes):
# 		ax.imshow(img)
# 		ax.axis('off')
# 	plt.tight_layout()
# 	plt.show()
#
# imgs , labels =train_batches[0]
# plotImages(imgs)
# print(labels[:5])
# sys.exit()

# scale an array of images to a new size

#Implement FID with Keras

#Load data with
#Load the training data

# prepare the inception v3 model
# model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# # load cifar10 images
# # print('Loaded', images1.shape, images2.shape)
# # convert integer to floating point values
# # images1 = images1.astype('float32')
# # images2 = images2.astype('float32')
# # resize images
# # images1 = scale_images(images1, (299,299,3))
# # images2 = scale_images(images2, (299,299,3))
# # print('Scaled', images1.shape, images2.shape)
# # # pre-process images
# # images1 = preprocess_input(images1)
# # images2 = preprocess_input(images2)
# # calculate fid
# fid = calculate_fid(model, images1, images2)
# print('FID: %.3f' % fid)

# # example of calculating the frechet inception distance
# import numpy
# from numpy import cov
# from numpy import trace
# from numpy import iscomplexobj
# from numpy.random import random
# from scipy.linalg import sqrtm
#
# # calculate frechet inception distance
# def calculate_fid(act1, act2):
# 	# calculate mean and covariance statistics
# 	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
# 	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
# 	# calculate sum squared difference between means
# 	ssdiff = numpy.sum((mu1 - mu2)**2.0)
# 	# calculate sqrt of product between cov
# 	covmean = sqrtm(sigma1.dot(sigma2))
# 	# check and correct imaginary numbers from sqrt
# 	if iscomplexobj(covmean):
# 		covmean = covmean.real
# 	# calculate score
# 	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
# 	return fid
#
# # define two collections of activations
# act1 = random(100*2048)
# act1 = act1.reshape((100,2048))
# act2 = random(100*2048)
# act2 = act2.reshape((100,2048))
# # fid between act1 and act1
# fid = calculate_fid(act1, act1)
# print('FID (same images): %.3f' % fid)
# # fid between act1 and act2
# fid = calculate_fid(act1, act2)
# print('FID (different images): %.3f' % fid)
