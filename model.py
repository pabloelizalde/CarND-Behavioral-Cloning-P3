import csv
import cv2
import numpy as np
import os
from sklearn.utils import shuffle

samples = []
with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
                samples.append(line)
        samples.pop(0) #remove first line with text
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
                
def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1:
                shuffle(samples) 
                for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset:offset+batch_size]
                
                        images,angles = get_data_from_batches(batch_samples)
                
                        X_train = np.array(images)
                        y_train = np.array(angles)
                        yield shuffle(X_train, y_train)
                
def get_data_from_batches(batch_samples): 
        for line in batch_samples:
                images = []
                measurements = []
                camera = np.random.choice(['center','left','right'])
                angle = float(line[3].strip('"')) 
                if camera == 'left':
                        source_path = line[1]
                        measurement = angle + 0.15 #correction
                elif camera == 'right':
                        source_path = line[2]
                        measurement = angle - 0.15 #correction
                else:
                        source_path = line[0]
                        measurement = angle

                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                image = cv2.imread(current_path)

                images.append(image)
                measurements.append(measurement)
                images.append(cv2.flip(image,1))
                measurements.append(measurement*-1.0)

        return images,measurements


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/32, validation_data=validation_generator, validation_steps=len(validation_samples)/32, epochs=3)

model.save('model.h5')