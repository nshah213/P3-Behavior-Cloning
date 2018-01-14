
import csv
import cv2
import numpy as np

samples = []
with open('./NewData/recorrected_data_set2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

measurements =[]
with open('./NewData/recorrected_measurements2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		measurements.append(str(line[0]))

#for i in range (len(measurements)):
#	samples[i][3] = measurements[i]

print(str(samples[0][3]))

#with open('./NewData/selected_wNewMeasurements.csv', 'w') as csvfile:
#	linewriter = csv.writer(csvfile, delimiter=',')
#	for line in samples:
#		linewriter.writerow(line)

measurements = np.array(measurements, dtype = float)
sorted_samples = []
sort_args = np.argsort(measurements)

print(sort_args)

for i in range(len(measurements)):
	current = samples[sort_args[i]]
	sorted_samples.append(current)

sorted_measurements = measurements[sort_args]
#sorted_samples = samples[[sort_args]]
last_measurement = 0.
idx = 0
measurements = []
lines = []
for sample in sorted_samples:
	image = cv2.imread(sample[0])
	measurement = sorted_measurements[idx]
	xLen = (np.shape(image)[0])
	croppedImage = image[65:xLen-25,:,:]
	#yuvImage = cv2.Color(croppedImage, cv2.COLOR_BGR2YUV)
	print(measurement)
	debug_on = True
	new_measurement = measurement
	idx += 1
	while(debug_on):
		cv2.imshow('ROI',croppedImage)
		key = cv2.waitKey(1) & 0xFF
		if (key == ord("a")):
			lines.append(np.array(sample))
			measurements.append(new_measurement)
			last_measurement = new_measurement
			
			debug_on = False
		if (key == ord("d")):
			debug_on = False
		if (key == ord("i")):
			new_measurement = new_measurement * 1.1
			if new_measurement >= 1.0:
				new_measurement = 1.0
			if new_measurement <= -1.0:
				new_measurement = -1.0
			print("New measurement = "+ str(new_measurement))
		if (key == ord("p")):
			new_measurement = new_measurement * 0.9
			print("New measurement = "+ str(new_measurement))
		if (key == ord("v")):
			new_measurement = new_measurement * -1
			print("New measurement = "+ str(new_measurement))
		if (key == ord("k")):
			new_measurement = last_measurement
			print("Old measurement restored = "+ str(new_measurement))
	#uncomment for debug mode to test change on limited set first
	#if (idx > 10): 
	#	break

with open('./NewData/recorrected_data_set3.csv', 'w') as csvfile:
	linewriter = csv.writer(csvfile, delimiter=',')
	for line in lines:
		linewriter.writerow(line)

outfile = open('./NewData/recorrected_measurements3.csv', 'w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], measurements))
outfile.close()

