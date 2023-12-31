import numpy as np
import tensorflow
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, save_model
from sklearn.metrics import classification_report

image_width, image_height = 224, 224
batch_size = 32
num_classes = 10

class ImageClassifier:
    def __init__(self):
        self.model = None
        self.train_data = None
        self.val_data = None
        self.image_width = image_width
        self.image_height = image_height
        self.batch_size = batch_size
        self.num_classes = num_classes

    def load_data(self, train_directory, val_directory):
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        self.train_data = train_datagen.flow_from_directory(
            directory=train_directory,
            target_size = (self.image_width, self.image_height),
            batch_size=self.batch_size,
            class_mode = "categorical",
            shuffle = True,
            seed = 42
        )
        self.val_data = val_datagen.flow_from_directory(
            directory=val_directory,
            target_size=(self.image_width, self.image_height),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )

    def create_model(self):
        net = keras.applications.mobilenet_v2.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(self.image_width, self.image_height, 3)
        )
        for layer in net.layers[:-2]:
            layer.trainable = False
        x = layers.GlobalAveragePooling2D()(net.output)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation="softmax")(x)
        self.model = Model(inputs=net.input, outputs=output)
        self.model.summary()

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

    def train_model(self, epochs):
        self.model.fit(
            self.train_data,
            epochs=epochs,
            validation_data=self.val_data
        )

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.val_data, verbose=2)
        print(f"Accuracy: {accuracy}, Loss: {loss}")

    def save_model(self):
        models.save_model(self.model, 'model/TomatoLeafMobilenet.h5')

    def testing_model(self):
        test_pred = self.model.predict(self.val_data, verbose=1)
        test_labels = np.argmax(test_pred, axis=1)
        class_labels = self.val_data.class_indices
        class_labels = {v: k for k, v in class_labels.items()}
        classes = list(class_labels.values())
        print(classes)
        print('Classification Report')
        print(classification_report(self.val_data.classes, test_labels, target_names=classes))

print("******************************************************************************************")
print("Creating the class")
print("******************************************************************************************")
classifier = ImageClassifier()
print("******************************************************************************************")
print("Getting the image data")
print("******************************************************************************************")
train_directory = "/Users/dipit/Image Data/Tomato/Data/train"
val_directory = "/Users/dipit/Image Data/Tomato/Data/val"
classifier.load_data(train_directory, val_directory)
print("******************************************************************************************")
print("Creating the model")
print("******************************************************************************************")
classifier.create_model()
print("******************************************************************************************")
print("Training")
print("******************************************************************************************")
epochs = 10
classifier.train_model(epochs)
print("******************************************************************************************")
print("Evaluating")
print("******************************************************************************************")
classifier.evaluate_model()
print("******************************************************************************************")
print("Saving Model")
print("******************************************************************************************")
classifier.save_model()
print("Model Saved..")
print("******************************************************************************************")
print("Testing and Prediction Report")
print("******************************************************************************************")
classifier.testing_model()
