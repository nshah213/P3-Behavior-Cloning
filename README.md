**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./images_writeup/preprocessing_and_augmentation_pipeline.png "Preprocessing and augmentation"
[image3]: ./images_writeup/distribution_of_driving_data.png "Distribution of dataset"
[image4]: ./images_writeup/distribution_of_balanced_data.png "Distribution of balanced dataset"


#### Description
 
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. It contains of the following sections - 
1. Generator for loading small parts of training and validation data in to memory at a given time. This is contained in lines 137 to 213. 
2. Definition of the convolutional neural net using keras, present in code lines 221 to 254. 
3. Setting up the loss fucntion and the optimizer to be used, present in lines 257, 258
4. Code for training the model and saving trained model, lines 260 to 265 for intial training and lines 270 t0 275 for further training of the net using a lower learning rate
5. Code to visualize the preprocessing and data augmentation pipeline is available on lines 47 to 133. Uncomment this only for documentation purposes and leave commented to train the model


### Model Architecture and Training Strategy

#### 1. Model architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64 (model.py lines 226-253). The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

As part of the preprocessing all the training set images were cropped to same height but with a small random offset. Thus, everytime a particular image is input again in to the neural net it is slightly different from the previous version of the image to avoid overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was collected by manually driving the vehicle in the simulation environment provided by Udacity for the purpose of the project. Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Further, I have combined data from the 2 tracks to see if we are able to train the same neural net to navigate both the tracks in the simulator. I am aware that it has been advised not to merge the datasets, but I wanted to experiment and have got something that works well for the basic track and can complete large parts of the challenge track on its own. 

Currently, most times when the model is not able to control the vehicle in center of the road can be attributed to a noisy data in the training set. Noisy data in this context is defined as commanded steering angle such that it would result in loss of control of the vehicle.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a known solution that is well suited for similar problem and then modify it to make the project successful. I used CNN architecture published by Nvidia as my starting point. Next, I have simplified the model greatly by using smaller convolutional filter size, reducing the number of channels for the convolutional layers and reducing the total number of neurons in the fully connected layers. This is because, the simulator track environment is far less complex than real world driving scenario.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

During the training phases even before running on simulator, it became evident that the model was not able to fit better than what roughly translated to an mean error of 0.3 in prediction, which is 15% of entire steering command range from maximum left to maximum right.

I created a script to load each of the recorded images in the file and print the training steering command and was shocked by the quality of signal data for driving recorded with a keyboard of purposes of training a neural net. This was because the keyboard drving behavior tended to be more on/off control rather than a continuous control setpoint that the neural net would be expected to predict for any given driving scenario. 

The next version of training data with mouse as input instead to see if I could get a good continuous response.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases and in most cases it almost always resulted from poor training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is the summary for the final model architecture. 

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 70, 320, 3)    0           lambda_input_2[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 68, 318, 16)   448         lambda_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 68, 318, 16)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
averagepooling2d_1 (AveragePooli (None, 34, 159, 16)   0           activation_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 157, 32)   4640        averagepooling2d_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 157, 32)   0           convolution2d_2[0][0] 
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 16, 78, 32)    0           activation_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 14, 76, 64)    18496       averagepooling2d_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 14, 76, 64)    0           convolution2d_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 7, 38, 64)     0           activation_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 17024)         0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 600)           10215000    flatten_1[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 600)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           60100       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 100)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 25)            2525        activation_5[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 25)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             26          activation_6[0][0]
____________________________________________________________________________________________________

Total params: 10,301,235
Trainable params: 10,301,235
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the sides if it drifts away from center line. 

Then I repeated this process on track two in order to get more data points.

For data augmentation, I used the images captured from the left and the right cameras also with an offset in the commanded steering angle for those. Following is the visualization of the data augmentation and preprocessing steps - 
![alt text][image2]

After the collection process, I had 31655 data points for the images captured from the central camera alone. Following the visualization of the distribution of those points - 
![alt text][image3]

I wrote a script to create a randomly selected balanced dataset. This reduced the dataset to 7427 images. This is what the data distribution looked like after balancing.
![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 25-30 as evidenced by the error on validation set not decrreasing with further reduction error in the training set. 

I used an adam optimizer so that manually training the learning rate wasn't necessary. However, I did reduce the learning rate and retrain for another 20 epochs to finetune the CNN further.

 
