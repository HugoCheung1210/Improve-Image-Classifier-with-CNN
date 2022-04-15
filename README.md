# Improve-MNIST-with-CNN
Improve MNIST to 99.5% accuracy or more by adding only a single convolutional layer and a single MaxPooling 2D layer to the model

You should stop training once the accuracy goes above this amount. It should happen in less than 10 epochs, so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your callback.

When 99.5% accuracy has been hit, you should print out the string "Reached 99.5% accuracy so cancelling training!"

Begin by loading the data. A couple of things to notice:

The file mnist.npz is already included in the current workspace under the data directory. By default the load_data from Keras accepts a path relative to ~/.keras/datasets but in this case it is stored somewhere else, as a result of this, you need to specify the full path.

load_data returns the train and test sets in the form of the tuples (x_train, y_train), (x_test, y_test) but in this exercise you will be needing only the train set so you can ignore the second tuple.

One important step when dealing with image data is to preprocess the data. During the preprocess step you can apply transformations to the dataset that will be fed into your convolutional neural network.

Here you will apply two transformations to the data:

Reshape the data so that it has an extra dimension. The reason for this is that commonly you will use 3-dimensional arrays (without counting the batch dimension) to represent image data. The third dimension represents the color using RGB values. This data might be in black and white format so the third dimension doesn't really add any additional information for the classification process but it is a good practice regardless.
Normalize the pixel values so that these are values between 0 and 1. You can achieve this by dividing every value in the array by the maximum.
Remember that these tensors are of type numpy.ndarray so you can use functions like reshape or divide to complete the reshape_and_normalize function

Expected Output of the reshape_and_normalize function:

Maximum pixel value after normalization: 1.0

Shape of training set after reshaping: (60000, 28, 28, 1)

Shape of one image after reshaping: (28, 28, 1)


If you see the message that you defined in your callback printed out after less than 10 epochs it means your callback worked as expected.
