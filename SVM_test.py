import sys
import os



import numpy as np
import mnist_loader
import mnist
from sklearn import svm



training_data, validation_data, test_data = mnist_loader.load_data()

images_training=mnist.get_images(training_data)
images_true=training_data[1]

images_test=mnist.get_images(test_data)


mnist.plot_mnist_digit(images_test[0])



#SVM test
span=1000

training_sel_nums=training_data[0][:span]
training_sel_vals=training_data[1][:span]


# train
clf = svm.SVC()
clf.fit(training_sel_nums, training_sel_vals)

# test
predictions = [int(a) for a in clf.predict(test_data[0])]
num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))

print "%s of %s values correct." % (num_correct, len(test_data[1]))




predictions[:10]
test_data[1][:10]

mnist.plot_images_separately(images_test[:10])



clf.predict(test_data[0])


