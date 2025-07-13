# ✅ Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ✅ Load and Normalize Data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# ✅ Data Augmentation Setup
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train.reshape(-1, 28, 28, 1))  # Required for grayscale images

# ✅ 🔍 Visualize Augmented Samples
sample_image = X_train[0].reshape(1, 28, 28, 1)
aug_iter = datagen.flow(sample_image, batch_size=1)

plt.figure(figsize=(6, 6))
for i in range(9):
    aug_image = next(aug_iter)[0].reshape(28, 28)
    plt.subplot(3, 3, i+1)
    plt.imshow(aug_image, cmap='gray')
    plt.axis('off')
plt.suptitle("🌀 Augmented MNIST Samples", fontsize=14)
plt.tight_layout()
plt.show()

# ✅ CNN Model Definition
model = keras.Sequential([
    keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# ✅ Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train Model with Augmentation
history = model.fit(
    datagen.flow(X_train.reshape(-1, 28, 28, 1), y_train, batch_size=64),
    epochs=10,
    validation_data=(X_test.reshape(-1, 28, 28, 1), y_test)
)

# ✅ Evaluate Model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
print("✅ CNN with Augmentation Test Accuracy:", test_acc)

# ✅ 🔍 Visualize Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('📈 Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('📉 Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
# ✅ Final Evaluation
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
print("✅ Final Test Accuracy:", test_acc)  

# ✅ Save the Trained Model
model.save("mnist_cnn_model.h5")
print("✅ Model saved as mnist_cnn_model.h5")
