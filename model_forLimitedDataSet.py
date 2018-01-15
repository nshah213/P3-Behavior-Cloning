import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

csvFile = './data/driving_log.csv'
samples = []
with open(csvFile) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

#measurements =[]
#with open('./NewData/recorrected_measurements3.csv') as csvfile:
#	reader = csv.reader(csvfile)
#	for line in reader:
#		measurements.append(str(line[0]))

print(np.shape(samples))
print(samples[0])
#for i in range (len(measurements)):
#	samples[i][3] = measurements[i]

#print(str(samples[0][3]))
"""
with open('./NewData/selected_wNewMeasurements.csv', 'w') as csvfile:
	linewriter = csv.writer(csvfile, delimiter=',')
	for line in samples:
		linewriter.writerow(line)
"""
ON_AWS = True

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot

###For visualization - uncomment only for documentation
"""
for batch_sample in samples:
	name_center = './NewData/IMG/'+batch_sample[0].split('/')[-1]
	name_left = './NewData/IMG/'+batch_sample[1].split('/')[-1]
	name_right = './NewData/IMG/'+batch_sample[2].split('/')[-1]

	crop_bias = int(np.random.random()*6)
	#print(name)				
	center_image = cv2.imread(name_center)
	#print(batch_sample[3][:])
	center_angle = float(batch_sample[3][:])*1.1
	if ((center_angle < 0.02) & (center_angle > -0.02)):				
		center_angle *= 1.0
	if ((center_angle < 0.25) & (center_angle > 0.02)):				
		center_angle *= 1.0
	if ((center_angle < -0.02) & (center_angle > -0.2)):				
		center_angle *= 1.0
	if ((center_angle < 0.4) & (center_angle > 0.1)):				
		center_angle *= 1.1
		center_angle = min(center_angle,1.0)
	if ((center_angle < -0.1) & (center_angle > -0.4)):				
		center_angle *= 1.0
		center_angle = max(center_angle,-1.0)
	#if ((center_angle < 0.4) & (center_angle > 0.2)):				
	#	center_angle *= 1.2
	#if ((center_angle < -0.2) & (center_angle > -0.4)):
	#if leftData:
	left_image = cv2.imread(name_left)
	left_angle = center_angle + 0.25

	if left_angle > 1.0:
		left_angle = 1.0
	
	right_image = cv2.imread(name_right)
	right_angle = center_angle - 0.25
	if right_angle < -1.0:
		right_angle = -1.0
	xLen = 160
	left_cropped = left_image[65:xLen-25,:,:]
	right_cropped = right_image[65:xLen-25,:,:]				
	croppedImage = center_image[62 + crop_bias:xLen + crop_bias - 28,:,:]
			
	image_flipped = np.fliplr(croppedImage)
	left_flipped = np.fliplr(left_cropped)              
	right_flipped = np.fliplr(right_cropped)              
	document = True
	if document:			
		f, ax = plt.subplots(3,3,figsize = (9,4.7))
		print(np.shape(ax))
		ax[0,0].imshow(np.flip(left_image,2))
		ax[0,0].axis('off')
		ax[0,0].set_title("Left camera")

		ax[0,1].imshow(np.flip(center_image,2))
		ax[0,1].axis('off')
		ax[0,1].set_title("Center camera")

		ax[0,2].imshow(np.flip(right_image,2))
		ax[0,2].axis('off')
		ax[0,2].set_title("Right camera")
			
		ax[1,0].imshow(np.flip(left_cropped,2))
		ax[1,0].axis('off')
		ax[1,0].set_title("Left cropped")

		ax[1,1].imshow(np.flip(croppedImage,2))
		ax[1,1].axis('off')
		ax[1,1].set_title("Center cropped")


		ax[1,2].imshow(np.flip(right_cropped,2))
		ax[1,2].axis('off')
		ax[1,2].set_title("Right cropped")
		
		ax[2,0].imshow(np.flip(left_flipped,2))
		ax[2,0].axis('off')
		ax[2,0].set_title("Left flipped")

		ax[2,1].imshow(np.flip(image_flipped,2))
		ax[2,1].axis('off')
		ax[2,1].set_title("Center flipped")

		ax[2,2].imshow(np.flip(right_flipped,2))
		ax[2,2].axis('off')
		ax[2,2].set_title("Right flipped")

		plt.tight_layout()
		plt.show()
"""

def generator(samples, batch_size=32):
	num_samples = len(samples)
	print(len(samples))
	while 1: # Loop forever so the generator never terminates
		samples = shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			leftData = True
			for batch_sample in batch_samples:
				name_center = './data/IMG/'+batch_sample[0].split('/')[-1]
				name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
				name_right = './data/IMG/'+batch_sample[2].split('/')[-1]
				
				crop_bias = int(np.random.random()*6)
				#print(name)				
				center_image = cv2.imread(name_center)
				#print(batch_sample[3])
				center_angle = float(batch_sample[3])
				left_image = cv2.imread(name_left)
				left_angle = center_angle + 0.1
				#print("Left")
				if left_angle > 1.0:
					left_angle = 1.0
				#leftData = False
				#else:
				right_image = cv2.imread(name_right)
				right_angle = center_angle - 0.1
				if right_angle < -1.0:
					right_angle = -1.0
				#print("Right")
				#leftData = True
				xLen = 160
				left_cropped = left_image[65:xLen-25,:,:]
				right_cropped = right_image[65:xLen-25,:,:]				
				#print(np.shape(center_image))
				croppedImage = center_image[62 + crop_bias:xLen + crop_bias - 28,:,:]
				
				image_flipped = np.fliplr(croppedImage)              
				images.append(croppedImage)
				images.append(image_flipped)
				angles.append(center_angle)
				angles.append(center_angle * -1.)

				left_flipped = np.fliplr(left_cropped)              
				images.append(left_cropped)
				images.append(left_flipped)
				angles.append(left_angle)
				angles.append(left_angle * -1.)
				right_flipped = np.fliplr(right_cropped)              
				images.append(right_cropped)
				images.append(right_flipped)
				angles.append(right_angle)
				angles.append(right_angle * -1.)
				
			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			#print(len(X_train))
			yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 70, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2)))
#model.add(Dropout(0.15))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(AveragePooling2D((2, 2)))
#model.add(Dropout(0.1))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.05))

#model.add(Convolution2D(128,3,3))
#model.add(Activation('relu'))
#model.add(Dropout(0.05))

model.add(Flatten())
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dense(50))
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dense(1))

#model.compile('adam', 'mean_squared_error')
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer= adam)

model.summary()

total_epochs = 30
for i in range(total_epochs):
	model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples)*6, validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=1)
	model.save('./all_files/modelA'+str(i)+'.h5')

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer= adam)

total_epochs = 30
for i in range(total_epochs):
	model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples)*6, validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=1)
	model.save('./all_files/modelB'+str(i)+'.h5')

#/model = load_model('modelB8_19.h5')
#plot(model, to_file='model.png')

