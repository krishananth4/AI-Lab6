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
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the pixel values
x_train_full, x_test = x_train_full / 255.0, x_test / 255.0

# Flatten the images
x_train_full = x_train_full.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Split the full training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x_train_full, y_train_full, stratify=y_train_full, random_state=2, test_size=0.25)

# Create the neural network model
model = Sequential()
model.add(Input(shape=(28*28,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Predict the classes
predictions = model.predict(x_test)
y_test_hat = np.argmax(predictions, axis=1)

# Calculate the accuracy rate
num_correct = np.sum(y_test_hat == y_test)
accuracy_rate = num_correct / len(y_test)
print(f"Accuracy Rate = {accuracy_rate}")

# Display the confusion matrix
fashion_mnist_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
conf_matrix = confusion_matrix(y_test, y_test_hat)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=fashion_mnist_labels, yticklabels=fashion_mnist_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Display one image from each class
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i+1)
    class_indices = np.where(y_train_full == i)[0]
    class_image = x_train_full[class_indices[0]].reshape(28, 28)
    plt.imshow(class_image, cmap='gray')
    plt.title(fashion_mnist_labels[i])
    plt.axis('off')
plt.show()
