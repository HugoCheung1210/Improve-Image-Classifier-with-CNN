import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the data

## In case if the dataset is already in your working space
# Get current working directory
current_dir = os.getcwd() 

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz") 

# Get only training set
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

## In case you need to download the dataset MNIST
# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()


def reshape_and_normalize(images):
   
    # Reshape the images to add an extra dimension
    images = images.reshape((60000,28,28,1))
    
    # Normalize pixel values
    images = np.divide(images,np.max(images))

    return images
    
# Reload the images in case you run this cell multiple times
(training_images, training_labels), (x_test, y_test)= tf.keras.datasets.mnist.load_data(path=data_path) 


# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') is not None and logs.get('accuracy') > 0.995):
                print("\nReached 99.5% accuracy so cancelling training!") 
                
                # Stop training once the above condition is met
                self.model.stop_training = True
                
                
def convolutional_model():
    # Define the model, it should have 5 layers:
    # - A Conv2D layer with 32 filters, a kernel_size of 3x3, ReLU activation function
    #    and an input shape that matches that of every image in the training set
    # - A MaxPooling2D layer with a pool_size of 2x2
    # - A Flatten layer with no arguments
    # - A Dense layer with 128 units and ReLU activation function
    # - A Dense layer with 10 units and softmax activation function
    model = tf.keras.models.Sequential([ 
     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(128,activation='relu'),
     tf.keras.layers.Dense(10,activation='softmax')
    ]) 

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
        
    return model
    
# Save your untrained model
model = convolutional_model()

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
