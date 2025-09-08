import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1. Unzip Dataset
# -----------------------------
zip_path = r"C:\Users\DELL\Documents\ML\Cats_Vs_Dog_Classifier\cats_and_dogs_filtered.zip"
extract_path = r"C:\Users\DELL\Documents\ML\Cats_Vs_Dog_Classifier"

if not os.path.exists(os.path.join(extract_path, "cats_and_dogs_filtered")):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted!")

base_dir = os.path.join(extract_path, "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# -----------------------------
# 3. Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 4. Train Model
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
loss, acc = model.evaluate(val_generator)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")

# -----------------------------
# 6. Predict Random Image from Validation
# -----------------------------
# Pick either 'cats' or 'dogs' folder
category = random.choice(["cats", "dogs"])
folder = os.path.join(val_dir, category)

# Pick a random image from that folder
random_image = random.choice(os.listdir(folder))
sample_path = os.path.join(folder, random_image)

# Load and preprocess the image
img = image.load_img(sample_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Model prediction
prediction = model.predict(img_array)
pred_label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

# Show the image with predicted label
plt.imshow(plt.imread(sample_path))
plt.title(f"Prediction: {pred_label}\nActual: {category.capitalize()}")
plt.axis("off")
plt.show()
