import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
# The dataset is loaded into training and testing sets
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the pixel values
x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

# Reshape the testing data
x_train_full = x_train_full.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Split the full training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x_train_full, y_train_full, stratify=y_train_full, random_state=2, test_size=0.25)

# Create the model
# We start by creating a Sequential Model
model = Sequential()
# We create the input layer of the neural network
model.add(Input(shape=(28*28,)))
# We create the hidden layer that has 512 nodes and uses the ReLU activation function 
model.add(Dense(512, activation='relu'))
# We create the output layer tha has 10 nodes and uses the softmax activation function
model.add(Dense(10, activation='softmax'))

# We compile the model implementing the loss function 'sparse_categorical_crossentropy', the optimizer as 'adam', and the metrics as ['accuracy]
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# We input our training data and train it using 3 epochs with a batch size of 32
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)

# Using testing data we will try to make prediction with it
predictions = model.predict(x_test)
# the predicted data is turned into class labels
# the model tries to predict this variable
y_test_hat = np.argmax(predictions, axis=1)

# Calculate the accuracy rate
# First, we check to see number of correct predictions we got
num_correct = np.sum(y_test_hat == y_test)
# We calculate the accuracy rate by dividing the number of correct predictions over the total number of data in the y-axis of the dataset we are testing
accuracy_rate = num_correct / len(y_test)
# We print out the final accuracy rate after we compelete 5 epochs
print(f"Accuracy Rate = {accuracy_rate}")

# The confusion matrix is displayed using matplotlib and seaborn
# The labels for the confusion matrix is created
fashion_mnist_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Using the testing data we calculate the confusion matrix using the scikit learn library
conf_matrix = confusion_matrix(y_test, y_test_hat)
# We set the dimensions for the confusion matrix display
plt.figure(figsize=(9, 7))
# We use the seaborn library to create a heatmap that can be displayed in the confusion matrix as it is displayed
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=fashion_mnist_labels, yticklabels=fashion_mnist_labels)
# We set the label for the x-axis of the confusion matrix
plt.xlabel('Predicted Label')
# We set the label for the y-axis of the confusion matrix
plt.ylabel('True Label')
# We set the title for the confusion matrix
plt.title('Confusion Matrix')
# We display the confusion matrix
plt.show()

# We set the dimensions for the image display
plt.figure(figsize=(7, 7))
# We iterate through all of the images that we can display
for i in range(10):
    # We format the images in a way where we can display it as a table with 5 rows and 2 columns
    plt.subplot(5, 2, i+1)
    # We check for the indices for wherever the y_train_full data is equal to the current clothing class we are in. We want to access the first index of the tuple
    class_indices = np.where(y_train_full == i)[0]
    # We reshape the image here to make it more compatible with the output window
    class_image = x_train_full[class_indices[0]].reshape(28, 28)
    # We display the images in grayscale
    plt.imshow(class_image, cmap='gray')
    # We set the title to whatever clothing class we are iterating over at this instant of the loop
    plt.title(fashion_mnist_labels[i])
    # We turn off the option for axis labels, which is usually needed when you format this as a subplot
    plt.axis('off')
# We display the subplot
plt.show()
