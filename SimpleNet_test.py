

import numpy as np
import mnist_loader
import mnist
import matplotlib.pyplot as plt



# Activation functions for neurons
def linear(z): return z
def ReLU(z): return np.max(0.0, z)






import keras
# Import EarlyStopping
from keras.callbacks import EarlyStopping

from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist







training_data, validation_data, test_data = mnist_loader.load_data()





# Specify the model

batch_size = 128
num_classes = 10
epochs = 12
span=50000

train_nums=training_data[0][:span]
train_vals=keras.utils.to_categorical(training_data[1][:span], num_classes)








n_cols = train_nums.shape[1]

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(784,activation="relu",input_shape=(784,)))
model.add(Dense(100,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(100,activation="relu"))

# Add the output layer
model.add(Dense(10,activation="softmax"))


# Compile the model
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])


# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
fit=model.fit(train_nums,train_vals,validation_split=0.3,epochs=20,callbacks=[early_stopping_monitor])


# Verify that model contains information from compiling
print("Loss function: " + model.loss)






# Create the plot
plt.plot(fit.history['val_acc'], 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.show()



# Calculate predictions: predictions
predictions = model.predict(test_data[0])

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)