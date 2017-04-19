#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model arquitecture of my neural network is basically the NVIDIA Architecture. That basically consists, first in a normalization layer, then five convolutional layers, the first three with filters of 5x5, and the last two with filters of 3x3. After each convolutional layer , there is a activation layer using RELU.
After that, there are four fully connected layers. In the second layer we also use Cropping2D to remove the unnecesary parts of the image. 

####2. Attempts to reduce overfitting in the model

During the development I included 50% dropout after each fully connected layer. But the result was not improving. I decided to remove the dropout layers.

To train my model I used different data sets spliting off 20% of the data to use for validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

####4. Appropriate training data

I use as a training data the one provided in the udacity website. After trying to record my own data (I was a terrible driver), the best results was with the Udacity data. For some cases, where the car was not driving very smooth, I increase the data with some of my own.  

###Model Architecture and Training Strategy

####1. Solution Design Approach

My strategy was starting for an easy architecture as LeNet is. The performance with this architecture was very poor, and the car went to one of the side almost after start driving. As mention in the videos, and after checking in the forum I changed my architecture to NVIDIA Architecture.

To be able to check that the model worked correctly and I split the data in training and validation. In both cases the squared error went down for the first epochs, but increase after the fourth epoch. I decided to decreased the epochs to two at the beggining with pretty good results for my model. But finally increased the value to three with an overall better result in the simulator.

Since I descarted the dropout for my model, I was worried about the overfitting of my model. Apart from that, in some specific spots of the track (bridge, dirty road) the car was felling off the track. To fix these two issues, I decided to record more data manually for those spots where the car was not driving so well. After append this new data to the previous one, the error of my model and the driving improved.
 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I left the car driving autonomously during 2 hours aprox., and was the whole time on track. 

####2. Final Model Architecture

The final model architecture (model.py lines 68-79) is the NVIDIA architecture, with a cropping layer to remove the useless parts of the images. 

####3. Creation of the Training Set & Training Process

The best results for my model I obtained them with the Udacity data. As mentioned before, the car drove not so good in specific spots of the track. So I decided to increment the data with my own. I then recorded the car recovering for those specific spots. 

As it can be seen in the method `get_data_from_batches` (model.py lines 30-55), for every image in data I randomly chose one (left, right or center) and adapt the angle accordingly to the camera. I also included a flipped version of that image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the training loss and the validation loss getting smaller and decreasing in each epoch.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
