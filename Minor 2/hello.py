# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

# %%
# Function to create train and validation generators
def create_generators(train_data_dir, valid_data_dir, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(255, 255),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = valid_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(255, 255),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator

# %%
# Define directories for training and validation data
train_data_dir = "archive (5)/Acne/Train"
valid_data_dir = "archive (5)/Acne/Validation"

# %%
# Create data generators
batch_size = 100
train_generator, validation_generator = create_generators(train_data_dir, valid_data_dir, batch_size)


# %%
# Define MobileNet without including the top layer
input_shape = (255, 255, 3)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)


# %%
# Freeze MobileNet layers
for layer in base_model.layers:
    layer.trainable = False


# %%
# Build your model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


# %%
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# %%
# Train the model
history = model.fit(train_generator,
                    epochs=5)


# %%
# Evaluate the model
true_classes = validation_generator.classes
predicted_classes = model.predict(validation_generator).flatten().round()


# %%
true_classes

# %%
predicted_classes

# %%
# Calculate evaluation metrics
precision, recall, f1_score, support = precision_recall_fscore_support(
    true_classes, predicted_classes, average='binary'
)

# %%
accuracy = accuracy_score(true_classes, predicted_classes)
conf_matrix = confusion_matrix(true_classes, predicted_classes)


# %%
# Print evaluation metrics
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1_score)
print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(conf_matrix)

# %%
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()