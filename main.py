import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


print(tf.__version__)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm


data = pd.read_csv("Movies-Poster_Dataset/train.csv")
img_width = 350
img_height = 350

X = []

for i in tqdm(range(data.shape[0])):
        path = "Movies-Poster_Dataset/Images/" + data["Id"][i] + ".jpg"
        img = image.load_img(path,target_size=(img_width,img_height,3))
        img = image.img_to_array(img)/255.0
        X.append(img)
X = np.array(X)

y = data.drop(['Id','Genre'],axis = 1)
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size = 0.15)

#Buidling the CNN

model = Sequential()
#First CNN Layer
model.add(Conv2D(16,(3,3),activation='relu',input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

#Second CNN Layer
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

#Third CNN Layer
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))

#Fourth CNN Layer
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

#First Fully connected layer
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Second Fully connected layer
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(25, activation='sigmoid'))



# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training/cp-{epoch:04d}.ckpt"


# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)


model.compile(optimizer="adam",loss="binary_crossentropy", metrics=['accuracy'])

history = model.fit(X_train,  
                    y_train, 
                    epochs=5, 
                    validation_data=(X_test,y_test),
                    callbacks=[cp_callback])

model.save('saved_model/workingModel')

#to Load the model:
# new_model = tf.keras.models.load_model('saved_model/my_model')