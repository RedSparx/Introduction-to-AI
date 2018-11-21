"""LABORATORY 8
EXERCISE #2: Train a neural network to recognize images from the Fashion-MNIST dataset.
    (a) Load the Fashion-MINST dataset of fashion articles from the Keras module.
    (b) Rescale the 8-bit values so that the pixel values are normalized between 0 and 1.
    (c) Plot the first 25 images from teh database with their labels.  Each is 25x25.
    (d) Create the neural network by defining it sequentially: Flat input layer, followed by two dense layers with 128
        and 10 units respectively.
    (e) Add activation functions: 'relu' and 'softmax' respectively.
    (f) Compile the model.  Use the 'adam' optimizer with a mean square error loss function.  Print a model summary.
    (g) The target vector labels are integers from 0 to 9.  Use the Keras to_categorical() utility function to turn
        this into a one hot vector.
    (h) Train the network over 100 epochs with a verbose output so that progress can be observed.
    (i) Plot the training accuracy with a title and axis labels.
    (j) Plot the first 25 testing images from the database.
    (k) Save the model structure in JSON format.
    (k) Save the model structure in YAML format.
    (k) Save the fully trained model.
"""

import os

import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
from keras.utils import to_categorical

# region Suppress CPU/GPU warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# endregion

# region Load the Fashion-MINST dataset of fashion articles from the Keras module, it there for testing.
# x are input images, y are output labels.  All images are 28x28 pixels
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# endregion

# region We will rescale the 8-bit values so that they are normalized between 0 and 1.
x_train_norm = x_train / 255
x_test_norm = x_test / 255
# endregion

#region Plot the first 25 images from the database with their labels.
figure1 = plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train_norm[i], cmap='gray_r')
    plt.xlabel(class_names[y_train[i]])
figure1.suptitle('First 25 Training Samples', size=16)
plt.show()
# endregion

# region Create the neural network.
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
# endregion

# region Train the network using a categorical target vector rather than the numerical labels.
y_train_cat = to_categorical(y_train)
Hist = model.fit(x_train_norm, y_train_cat, epochs=100, verbose=True)
# endregion

# region Plot the training accuracy history.
figure2 = plt.figure(figsize=(10, 10))
plt.plot(Hist.history['acc'])
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()
# endregion

# region Plot the first 25 images from the test database with their labels.
figure3 = plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 2)
Predictions = model.predict(x_test_norm)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test_norm[i], cmap='gray_r')
    plt.xlabel(class_names[Predictions[i].argmax()])
figure3.suptitle('First 25 Test Samples', size=16)
plt.show()
# endregion

# region Save model to YAML and JSON file formats.  Save the trained Keras model.
model_YAML = model.to_yaml()
with open(r'Data\Model.yaml', 'w') as json_file:
    json_file.write(model_YAML)

model_JSON = model.to_json()
with open(r'Data\Model.json', 'w') as json_file:
    json_file.write(model_JSON)

model.save(r'Data\Model.h5')
# endregion

