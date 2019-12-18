import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

import numpy as np
import os
from os import walk

class ImageRecognition:

    def __init__(self):
        self.__batch_size = 0
        self.__img_rows, self.__img_cols = 0, 0
        self.__num_of_classes = 0
        self.__samples_per_class = 0
        self.__total_samples = self.__num_of_classes * self.__samples_per_class
        self.__input_shape = (0, 0, 0)
        self.__x_train, self.__x_valid, self.__x_test = [], [], []
        self.__y_train, self.__y_valid, self.__y_test = [], [], []

        print("Empty Object of Class 'ImageRecognition' created..!")

    def initialise_parameters(self, para):
        self.__batch_size = para['batch_size']
        self.__epochs = para['epochs']
        self.__img_rows, self.__img_cols = para['img_rows'], para['img_cols']
        self.__num_of_classes = para['num_of_classes']
        self.__samples_per_class = para['samples_per_class']
        self.__total_samples = self.__num_of_classes * self.__samples_per_class
        self.__input_shape = (self.__img_rows, self.__img_cols, 1)

    def load_data(self, data_path):
        for (dirpath, dirnames, filenames) in walk(data_path):
            return filenames

    def prepare_data(self, data_path, filenames):
        i = 0
        x, y = [], []
        x_all, y_all = [], []
        for file in filenames:
            file_path = data_path + file
            x = np.load(file_path)
            x = x.astype('float32')  ##normalise images
            x /= 255.0
            y = [i] * len(x)  # create numeric label for this image

            x = x[:self.__samples_per_class]  # get our sample of images
            y = y[:self.__samples_per_class]  # get our sample of labels

            if i == 0:
                x_all = x
                y_all = y
            else:
                x_all = np.concatenate((x, x_all), axis=0)
                y_all = np.concatenate((y, y_all), axis=0)
            i += 1

        # split data arrays into  train and test segments
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

        x_train = x_train.reshape(x_train.shape[0], self.__img_rows, self.__img_cols, 1)
        self.__x_test = x_test.reshape(x_test.shape[0], self.__img_rows, self.__img_cols, 1)

        self.__input_shape = (self.__img_rows, self.__img_cols, 1)

        y_train = tf.keras.utils.to_categorical(y_train, self.__num_of_classes)
        self.__y_test = tf.keras.utils.to_categorical(y_test, self.__num_of_classes)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print("Per Class: ", self.__samples_per_class)
        print("Total: ", self.__total_samples)

        ## x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
        self.__x_train, self.__x_valid, self.__y_train, self.__y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

        print("\nData prepared successfully...!")

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.__input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.__num_of_classes, activation='softmax'))

        print("\nModel created successfully..!")
        return model

    def compile_model(self, model):
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=['accuracy'])
        print("\nModel compiled successfully..!")
        return model

    def train_model(self, model):
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir="./tb_log_dir", histogram_freq=0)]

        model.fit(self.__x_train, self.__y_train,
                  batch_size=self.__batch_size,
                  epochs=self.__epochs,
                  callbacks=callbacks,
                  verbose=1,
                  validation_data=(self.__x_valid, self.__y_valid))
        return model

    def evaluate_model(self, model):
        score = model.evaluate(self.__x_test, self.__y_test, verbose=1)

        print('\nTest loss:', score[0], 'Test accuracy:', score[1]*100)

    def predict_class(self, filenames, model):
        labels = [os.path.splitext(file)[0] for file in filenames]
        print(labels)
        print("\nFor each pair in the following, the first label is predicted, second is actual\n")
        for i in range(2):
            t = np.random.randint(len(self.__x_test))
            x1 = self.__x_test[t]
            x1 = x1.reshape(1, 28, 28, 1)
            p = model.predict(x1)
            print("--------------------------------------------------------------------")
            print("Predicted: ", labels[np.argmax(p)], "  Actual : ", labels[np.argmax(self.__y_test[t])])


# Create Object of class ImageRecognition
obj_rec = ImageRecognition()

# Load data
data_path = "data_files/" # Path to Dataset
filenames = obj_rec.load_data(data_path)

# Parameter setting
parameter_dict = {
    'batch_size' : 32,
    'img_rows' : 28,
    'img_cols' : 28,
    'num_of_classes' : len(filenames),
    'samples_per_class' : 1000,
    'epochs' : 3
}
# Call method to Initialise parameters
obj_rec.initialise_parameters(parameter_dict)

# Call method to prepare data for training, validation and testing
obj_rec.prepare_data(data_path, filenames)

# Call method to create a CNN Model
model = obj_rec.create_model()

# Call method to compile the CNN model with optimizers, loss and accuracy
obj_rec.compile_model(model)

# Call method to train the CNN model
model = obj_rec.train_model(model)

# Call method to evaluate the CNN model
obj_rec.evaluate_model(model)

# Call method to predict classes using CNN model
obj_rec.predict_class(filenames, model)

print("Successfully completed..! Jsn")