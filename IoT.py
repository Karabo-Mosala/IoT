#========================================================================================================================================================#
#                                                      This is an Artificial Intellegence (AI) Model                                                     #
#                                                                    developed by                                                                        #
#                                                                    Karabo Mosala                                                                       #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                       This is a simple case-based code using image classification model and Quantum Random Number Generator                            #
#                                                                for Internet of Things (IoT) Devices.                                                   #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                      A version of this code that provides flexibity to case-by-case is available,                                      #
#                                                                                                                                                        #
#                                                               Email: kaymosala99@gmail.com                                                             #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                                                                                                                                        #
#                                                                                                                                                        #
#========================================================================================================================================================#

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input shape and classes
input_shape = (96, 96, 3)  # Adjust the size to match your use case
num_classes = 10  # Change this based on the number of classes in your problem

# Create a base MobileNetV2 model (pre-trained on ImageNet, if available)
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data loading and preprocessing
# Prepare your dataset and split it into training, validation, and test sets

# Example data loading using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Normalize pixel values
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    'data/train',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,  # Adjust the number of epochs based on your problem
)

# Save the model for deployment
model.save('iot_model.h5')

