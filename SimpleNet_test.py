

import numpy as np
import mnist_loader
import mnist
import matplotlib.pyplot as plt


import keras
# Import EarlyStopping
from keras.callbacks import EarlyStopping

from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist as mnistdata







training_data, validation_data, test_data = mnist_loader.load_data()


#from keras.datasets import mnist

#training_data, test_data = mnistdata.load_data()


# Specify the model

batch_size = 128
num_classes = 10
epochs = 12
span=60000

train_nums=training_data[0][:span]
train_vals=keras.utils.to_categorical(training_data[1][:span], num_classes)




test_nums=validation_data[0]
test_vals=keras.utils.to_categorical(validation_data[1], num_classes)





n_cols = train_nums.shape[1]

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(784,activation="relu",input_shape=(784,)))
model.add(Dense(100,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))

# Add the output layer
model.add(Dense(10,activation="softmax"))


# Compile the model
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])


# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
fit=model.fit(train_nums,train_vals,
              batch_size=batch_size,
              verbose=1,
              validation_data=(test_nums, test_vals),
              epochs=epochs,
              callbacks=[early_stopping_monitor])


model.save('models/keras_784-100-100-50.h5')



# Create the plot
plt.plot(fit.history['val_acc'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.show()


# Calculate predictions: predictions
predictions = model.predict(test_data[0])


pred_vals=np.asarray([np.argmax(pred) for pred in predictions])
pred_sign=[ np.round(predictions[i,pred],2) for i,pred in enumerate(pred_vals)]

np.round(predictions[:10],2)


test_imgs=mnist.get_images(test_data)

pred_vals[:10]
pred_sign[:10]


mnist.plot_images_together(test_imgs[:10])

