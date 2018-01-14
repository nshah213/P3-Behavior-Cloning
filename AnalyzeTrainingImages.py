import csv
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

csvName = './NewData/driving_log.csv'
samples = []
with open('./NewData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

mes = []
with open('./NewData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        mes.append(float(line[3]))

sor_idx = np.argsort(mes)
sor = np.sort(mes)

cnt,bins = np.histogram(sor,20)
print(cnt)

uniform_sample = []
cumulative_cnt = 0
for j in range(len(cnt)):
	keep_probability = 600/cnt[j]
	for i in range(cumulative_cnt,cumulative_cnt+cnt[j]):
		chance = np.random.random()
		if chance < keep_probability:
			uniform_sample.append(samples[sor_idx[i]])
	cumulative_cnt += cnt[j]
print(len(uniform_sample))
#print(uniform_sample[100])

with open('./NewData/uniform_sample.csv', 'w') as csvfile:
	linewriter = csv.writer(csvfile, delimiter=',')
	for line in uniform_sample:
		linewriter.writerow(line)

#	for i in range(start,len(sor)):
uniform_mes = []
for line in uniform_sample:
	uniform_mes.append(float(line[3]))		

print(len(mes))

plt.hist(mes,20)
plt.title("Distribution of training set")
plt.xlabel("Commanded steering angle")
plt.ylabel("Number of images in data set")
plt.show()
cv2.namedWindow('ROI')

print(len(uniform_mes))

plt.hist(uniform_mes,20)
plt.title("Distribution of balanced dataset")
plt.xlabel("Commanded steering angle")
plt.ylabel("Number of images in data set")
plt.show()
cv2.namedWindow('ROI')


"""

lines = []
cam_id = 0
idx = 0

with open(csvName) as csvfile:
	reader = csv.reader(csvfile)
	#print(len(np.array(reader)))
	for line in reader:
		idx += 0
		source_path = line[cam_id]
		measurement = float(line[3])
		filename = source_path.split('/')[-1]
		current_path = './Data/IMG/' + filename
		image = cv2.imread(current_path)
		xLen = (np.shape(image)[0])	
		croppedImage = image[65:xLen-25,:,:]
		#yuvImage = cv2.Color(croppedImage, cv2.COLOR_BGR2YUV)
		print(measurement)
		debug_on = False	
		while(debug_on):	
			#cv2.imshow("ROI", image_contour)		
			cv2.imshow('ROI',croppedImage)
			key = cv2.waitKey(1) & 0xFF
			if (key == ord("a")):
				debug_on = False 
				lines.append(np.array(line))
			if (key == ord("d")):
				debug_on = False 
				#lines.append(line)
				
#linesArray = np.array(lines)

"""



"""
total = len(lines)
print(total)


print(len(lines_train))
print(len(lines_valid))

total = len(lines_train)
total_valid = len(lines_valid)

def rdAndPrpImage(img_idx,cam_id = 0):
	source_path = lines_train[img_idx][cam_id]
	measurement = float(lines_train[img_idx][3])
	filename = source_path.split('/')[-1]
	if ON_AWS == True:
		current_path = '../data/IMG' + filename
	else:
		current_path = './CollectedData/IMG/' + filename
	#print(current_path)
	image = cv2.imread(current_path)
	xLen = (np.shape(image)[0])	
	croppedImage = image[70:xLen-25,:,:]
	yuvImage = cv2.Color(croppedImage, cv2.COLOR_BGR2YUV)
	cv2.imshow('ROI,croppedImage')
	normImage = yuvImage / 255 - 0.5
	return [normImage, measurement]
		
def generateX_Y(size = 1024, iteration_num = 0):
	images = []
	measurements = []
	images_valid = []
	m_valid = []
	augmented_images = []
	augmented_measurements = []
	augmented_images_valid = []
	augmented_m_valid = []

	min_idx = 768*iteration_num
	max_idx = 768*(iteration_num+1)

	if max_idx > total:
		max_idx = total

	for i in range(min_idx,max_idx): # line in lines:
		
		#print(len(measurement))
		#print(measurement)
		
		
		blur = cv2.GaussianBlur(grayScale,(5,5),2)
		#image_edges = cv2.Canny(blur,60,100,apertureSize = 3, L2gradient = True)
		#print(np.shape(image_edges))
		#print(np.shape(image))
		image_edges = np.reshape(blur, (65,320,1))		
		image_final = np.concatenate((croppedImage, image_edges), axis = 2)		
		images.append(image_final)
		measurements.append(measurement)
		images.append(cv2.flip(image_final,1))
		measurements.append(measurement * -1.)
		
	#augmented_images, augmented_measurements = [],[]
#	for i in range(len(images)):
#		augmented_images.append(images[i])
#		augmented_measurements.append(measurements[i])
#		augmented_images.append(cv2.flip(images[i],1))
#		augmented_measurements.append(measurements[i] * -1)
	min_valid = 256*iteration_num
	max_valid = 256*(iteration_num +1)
	
	if max_valid > total_valid:
		max_valid = total_valid

	for i in range(min_valid,max_valid):
		source_path = lines_valid[i][0]
		measurement = float(lines_valid[i][3])
		filename = source_path.split('/')[-1]
		if ON_AWS == True:
			current_path = '../data/IMG' + filename
		else:
			current_path = './CollectedData/IMG/' + filename
		#print(current_path)
		image = cv2.imread(current_path)
		xLen = (np.shape(image)[0])	
		croppedImage = image[70:xLen-25,:,:]
		grayScale = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
		croppedImage = croppedImage / 255 - 0.5
		blur = cv2.GaussianBlur(grayScale,(5,5),2)
		#image_edges = cv2.Canny(blur,60,100,apertureSize = 3, L2gradient = True)
		#print(np.shape(image_edges))
		#print(np.shape(image))
		image_edges = np.reshape(blur, (65,320,1))		
		image_final = np.concatenate((croppedImage, image_edges), axis = 2)		
		images_valid.append(image_final)
		m_valid.append(measurement)
		images_valid.append(np.flip(image_final,1))
		m_valid.append(measurement * -1.)
	#augmented_images_valid, augmented_m_valid = [],[]
	#print(images_valid[0])	
	#print()
	#print("Here starts flipped")
#	print(cv2.flip((cv2.flip(images_valid[0],1)),1))
#	for i in range(len(images_valid)):
#		augmented_images_valid.append(images_valid[i])
#		augmented_m_valid.append(m_valid[i])
#		augmented_images_valid.append(cv2.flip(images_valid[i],1))
#		augmented_m_valid.append(m_valid[i] * -1)
	
	#print(np.dtype(np.array(images)))
	#print(np.dtype(augmented_images))	
	ret = [np.array(images), np.array(measurements), np.array(images_valid), np.array(m_valid)]
	yield ret



# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build Convolutional Pooling Neural Network with Dropout in Keras Here
model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(65, 320,4)))
model.add(AvgPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5))
model.add(Dropout(0.05))
model.add(Activation('relu'))
model.add(Convolution2D(6,5,5))
model.add(Dropout(0.05))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Dense(1))
model.compile('adam', 'mean_squared_error')

my_generator = generateX_Y()

bat_size = 1024
for j in range(10):
	for i in range(int(len(lines)/bat_size)):
				
		new_data = generateX_Y(bat_size, i)
		new_list = list(new_data)
		print(i)	
		print(len(new_list))	
		X_train = new_list[0][0]
		y_train = new_list[0][1]
		X_valid = new_list[0][2]
		y_valid = new_list[0][3]
	
		print(np.shape(X_train))
		print(np.shape(y_train))
		print(np.shape(X_valid))
		print(np.shape(y_valid))
		print(y_valid[0])
		# preprocess data
		print(y_train[0])
		history = model.fit(X_train, y_train, nb_epoch=1, validation_data=(X_valid,y_valid), batch_size = 128) #, 
		model.save('model_new.h5')
		#print(history.hist())
		#fig,ax = plt.subplots(1,1)
		#ax.plot(np.arange(len(y_train)),y_train)	
		#plt.show()

"""
