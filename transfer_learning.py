# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os

# Load the Dataset
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
data_dir = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
data_dir = os.path.join(os.path.dirname(data_dir), 'cats_and_dogs_filtered')

# Preprocessing the data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(160, 160),
    batch_size=32)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(160, 160),
    batch_size=32)

# Load the pre-trained model
pretrained_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                                      include_top=False,
                                                      weights='imagenet')

# Freeze the layers of the base model
pretrained_model.trainable = False

# Create the final model
model = models.Sequential([
    pretrained_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax')  # For two classes: cats and dogs
])

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=validation_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test with real images
from google.colab import files
from tensorflow.keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    img_path = fn  
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0  
    img_array = tf.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    predicted_class = 'Dog' if predictions[0][1] > predictions[0][0] else 'Cat'
    
    print(f"The image is a: {predicted_class}")
